import os
import tempfile
import zipfile

import numpy as np
import open3d as o3d
from PIL import Image


class MeshIO:
    """Handle loading and saving of 3D meshes and point clouds."""

    @staticmethod
    def load_mesh(path):
        """Load a supported 3D mesh as an Open3D TriangleMesh."""
        ext = os.path.splitext(path)[1].lower()

        if ext == ".obj":
            m = o3d.io.read_triangle_mesh(path, enable_post_processing=True)
            m.compute_vertex_normals()
            if not m.has_triangle_normals():
                m.compute_triangle_normals()
            return m

        elif ext in (".glb", ".gltf"):
            import trimesh

            raw = trimesh.load(path, force="scene")
            meshes = []
            for node_name in raw.graph.nodes_geometry:
                transform, geometry_name = raw.graph[node_name]
                tm = raw.geometry[geometry_name].copy()
                tm.apply_transform(transform)
                meshes.append(tm)

            if not meshes:
                raise RuntimeError("No mesh data found in GLB/GLTF.")

            texture_images = []
            mesh_uvs = []
            for tm in meshes:
                uv = getattr(tm.visual, "uv", None)
                material = getattr(tm.visual, "material", None)
                image = None
                if material is not None:
                    image = getattr(material, "baseColorTexture", None)
                    if image is None:
                        image = getattr(material, "image", None)

                if uv is None or image is None:
                    mesh_uvs.append(None)
                    texture_images.append(None)
                    continue

                mesh_uvs.append(np.asarray(uv, dtype=np.float64))
                texture_images.append(image.convert("RGB"))

            textured_images = [img for img in texture_images if img is not None]
            atlas = None
            atlas_regions = [None] * len(texture_images)
            if textured_images:
                atlas_width = max(img.width for img in textured_images)
                atlas_height = sum(img.height for img in textured_images)
                atlas = Image.new("RGB", (atlas_width, atlas_height))
                y_offset = 0
                for image_index, image in enumerate(texture_images):
                    if image is None:
                        continue
                    atlas.paste(image, (0, y_offset))
                    atlas_regions[image_index] = (
                        0,
                        y_offset,
                        image.width,
                        image.height,
                    )
                    y_offset += image.height

            vertices = []
            triangles = []
            triangle_uvs = []
            vertex_offset = 0
            has_complete_uvs = atlas is not None
            for tm, uv, region in zip(meshes, mesh_uvs, atlas_regions):
                mesh_vertices = np.asarray(tm.vertices, dtype=np.float64)
                mesh_faces = np.asarray(tm.faces, dtype=np.int32)
                vertices.append(mesh_vertices)
                triangles.append(mesh_faces + vertex_offset)
                vertex_offset += len(mesh_vertices)

                if uv is None or region is None:
                    has_complete_uvs = False
                    continue

                x_offset, y_offset, width, height = region
                corner_uvs = uv[mesh_faces.reshape(-1)].copy()
                corner_uvs[:, 0] = (
                    x_offset + corner_uvs[:, 0] * width
                ) / atlas.width
                corner_uvs[:, 1] = (
                    atlas.height
                    - y_offset
                    - height
                    + corner_uvs[:, 1] * height
                ) / atlas.height
                triangle_uvs.append(corner_uvs)

            m = o3d.geometry.TriangleMesh()
            all_vertices = np.concatenate(vertices, axis=0)
            all_triangles = np.concatenate(triangles, axis=0)
            m.vertices = o3d.utility.Vector3dVector(all_vertices)
            m.triangles = o3d.utility.Vector3iVector(all_triangles)

            if has_complete_uvs and triangle_uvs:
                all_triangle_uvs = np.concatenate(triangle_uvs, axis=0)
                m.triangle_uvs = o3d.utility.Vector2dVector(all_triangle_uvs)
                m.textures = [
                    o3d.geometry.Image(np.asarray(atlas, dtype=np.uint8))
                ]
                m.triangle_material_ids = o3d.utility.IntVector(
                    np.zeros(len(all_triangles), dtype=np.int32)
                )
                print(
                    f"Loaded {len(textured_images)} GLB material textures "
                    f"into a {atlas.width}x{atlas.height} atlas."
                )
            elif textured_images:
                raise RuntimeError(
                    "The GLB contains mixed textured and untextured geometry; "
                    "a complete texture atlas could not be created."
                )

            m.compute_vertex_normals()
            m.compute_triangle_normals()
            return m

        elif ext == ".usdz":
            from pxr import Usd, UsdGeom

            stage = Usd.Stage.Open(path)
            verts_all, faces_all, uvs_all = [], [], []
            off = 0

            for prim in stage.Traverse():
                if prim.GetTypeName() != "Mesh":
                    continue

                um = UsdGeom.Mesh(prim)
                pts = np.array(um.GetPointsAttr().Get(), dtype=np.float64)
                fidx = np.array(
                    um.GetFaceVertexIndicesAttr().Get(),
                    dtype=np.int32,
                )

                verts_all.append(pts)
                faces_all.append(fidx.reshape(-1, 3) + off)
                off += len(pts)

                st = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
                if st and st.IsDefined():
                    uv_vals = np.array(st.Get(), dtype=np.float64)
                    uv_idx = st.GetIndices()
                    if uv_idx is not None and len(uv_idx):
                        uvs = uv_vals[np.array(uv_idx, dtype=np.int32)]
                    else:
                        uvs = uv_vals
                    uvs_all.append(uvs.reshape(-1, 2))

            if not verts_all or not faces_all:
                raise RuntimeError("No mesh data found in USDZ.")

            all_v = np.concatenate(verts_all)
            all_f = np.concatenate(faces_all)

            m = o3d.geometry.TriangleMesh()
            m.vertices = o3d.utility.Vector3dVector(all_v)
            m.triangles = o3d.utility.Vector3iVector(all_f)

            if uvs_all:
                all_uv = np.concatenate(uvs_all)
                if len(all_uv) == len(all_f) * 3:
                    m.triangle_uvs = o3d.utility.Vector2dVector(all_uv)

                    with zipfile.ZipFile(path) as z:
                        cands = [
                            n
                            for n in z.namelist()
                            if n.lower().endswith(("_tex0.png", "_tex0.jpg"))
                        ]
                        if not cands:
                            cands = [
                                n
                                for n in z.namelist()
                                if n.lower().endswith(".png")
                                or n.lower().endswith(".jpg")
                            ]

                        if cands:
                            with tempfile.TemporaryDirectory() as tmp:
                                z.extract(cands[0], tmp)
                                img = Image.open(
                                    os.path.join(tmp, cands[0])
                                ).convert("RGB")
                                m.textures = [
                                    o3d.geometry.Image(
                                        np.array(img, dtype=np.uint8)
                                    )
                                ]
                                m.triangle_material_ids = o3d.utility.IntVector(
                                    np.zeros(len(all_f), dtype=np.int32)
                                )

            m.compute_vertex_normals()
            m.compute_triangle_normals()
            return m

        else:
            raise ValueError(f"Unsupported format: {ext}")

    @staticmethod
    def safe_name(name):
        """Sanitize a string for use as a filename or identifier."""
        keep = "-_."
        return "".join(
            c if (c.isalnum() or c in keep) else "_"
            for c in name
        )

    @staticmethod
    def load_obj_grouped(path):
        """Read grouped OBJ parts while preserving their UV mappings."""
        all_v = []
        all_vt = []
        groups = []
        cur_name = "model"
        cur_faces = []

        def flush():
            nonlocal cur_faces
            if cur_faces:
                groups.append((cur_name, cur_faces))
                cur_faces = []

        with open(path, "r", errors="ignore") as f:
            for line in f:
                tok = line.split()
                if not tok:
                    continue
                t = tok[0]
                if t == "v":
                    all_v.append(
                        [float(tok[1]), float(tok[2]), float(tok[3])]
                    )
                elif t == "vt":
                    all_vt.append([float(tok[1]), float(tok[2])])
                elif t in ("o", "g"):
                    flush()
                    cur_name = tok[1] if len(tok) > 1 else "unnamed"
                elif t == "f":
                    v_idx, uv_idx = [], []
                    for p in tok[1:]:
                        parts = p.split("/")
                        vi = int(parts[0])
                        vi = len(all_v) + vi if vi < 0 else vi - 1
                        v_idx.append(vi)
                        if len(parts) >= 2 and parts[1] != "":
                            ti = int(parts[1])
                            ti = len(all_vt) + ti if ti < 0 else ti - 1
                            uv_idx.append(ti)
                        else:
                            uv_idx.append(None)
                    for k in range(1, len(v_idx) - 1):
                        vtri = (v_idx[0], v_idx[k], v_idx[k + 1])
                        uvtri = (uv_idx[0], uv_idx[k], uv_idx[k + 1])
                        cur_faces.append((vtri, uvtri))

        flush()

        all_v = np.asarray(all_v, dtype=np.float64)
        all_vt = (
            np.asarray(all_vt, dtype=np.float64)
            if all_vt
            else np.zeros((0, 2))
        )

        if not groups:
            groups = [("model", [])]

        parts = []
        name_counts = {}
        for name, faces in groups:
            if not faces:
                continue

            base_name = MeshIO.safe_name(name) or "part"
            name_index = name_counts.get(base_name, 0)
            name_counts[base_name] = name_index + 1
            unique_name = (
                base_name
                if name_index == 0
                else f"{base_name}_{name_index + 1:02d}"
            )

            vtris = np.asarray([fc[0] for fc in faces], dtype=np.int64)
            uvtris = [fc[1] for fc in faces]

            used = np.unique(vtris.ravel())
            remap = {g: l for l, g in enumerate(used)}
            loc_v = all_v[used]
            loc_f = np.vectorize(remap.get)(vtris).astype(np.int32)

            part_has_uv = (
                len(all_vt) > 0
                and all(all(u is not None for u in tri) for tri in uvtris)
            )
            if part_has_uv:
                uv_index_flat = np.asarray(
                    [u for tri in uvtris for u in tri],
                    dtype=np.int64,
                )
                tri_uvs = all_vt[uv_index_flat]
            else:
                tri_uvs = None
                if len(all_vt) > 0:
                    print(
                        f"  WARN: Part '{name}' has incomplete UVs, "
                        "treating as un-UV'd."
                    )

            parts.append(
                {
                    "name": unique_name,
                    "vertices": loc_v,
                    "triangles": loc_f,
                    "triangle_uvs": tri_uvs,
                }
            )

        return parts

    @staticmethod
    def part_to_mesh(part):
        """Construct an Open3D TriangleMesh from a part dictionary."""
        m = o3d.geometry.TriangleMesh()
        m.vertices = o3d.utility.Vector3dVector(part["vertices"])
        m.triangles = o3d.utility.Vector3iVector(part["triangles"])
        if part.get("triangle_uvs") is not None:
            m.triangle_uvs = o3d.utility.Vector2dVector(part["triangle_uvs"])
        m.compute_vertex_normals()
        return m

    @staticmethod
    def write_multipart_obj(
        out_obj_path,
        baked_parts,
        tex_subdir="textures",
        make_zip=True,
        write_mesh_files=True,
    ):
        """Write a multipart OBJ with baked components and textures."""
        base = os.path.splitext(out_obj_path)[0]
        out_mtl_path = base + ".mtl"
        mtl_basename = os.path.basename(out_mtl_path)
        out_dir = os.path.dirname(out_obj_path) or "."
        tex_dir = os.path.join(out_dir, tex_subdir)
        os.makedirs(tex_dir, exist_ok=True)

        obj_lines = [f"mtllib {mtl_basename}"]
        mtl_lines = []
        v_off = 0
        vt_off = 0
        written = []

        for part in baked_parts:
            name = part["name"]
            mesh = part["mesh"]
            tex = part["tex"]
            verts = np.asarray(mesh.vertices)
            tris = np.asarray(mesh.triangles)
            tuvs = np.asarray(mesh.triangle_uvs)
            mat_name = f"mat_{name}"

            write_uv = len(tuvs) == len(tris) * 3 and tex is not None
            if tex is not None and len(tuvs) != len(tris) * 3:
                print(
                    f"  WARN: {name} has incorrect UV count, "
                    "writing without texture."
                )

            if write_uv:
                tex_name = f"{name}_texture.jpg"
                tex.convert("RGB").save(
                    os.path.join(tex_dir, tex_name),
                    quality=95,
                )
                written.append(os.path.join(tex_dir, tex_name))
                mtl_lines += [
                    f"newmtl {mat_name}",
                    "Ka 1.0 1.0 1.0",
                    "Kd 1.0 1.0 1.0",
                    "Ks 0.0 0.0 0.0",
                    "d 1.0",
                    "illum 2",
                    f"map_Kd {tex_subdir}/{tex_name}",
                    "",
                ]
            else:
                mtl_lines += [
                    f"newmtl {mat_name}",
                    "Kd 0.8 0.8 0.8",
                    "illum 1",
                    "",
                ]

            obj_lines.append(f"o {name}")
            obj_lines.append(f"g {name}")
            for v in verts:
                obj_lines.append(
                    f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}"
                )
            if write_uv:
                for uv in tuvs:
                    obj_lines.append(f"vt {uv[0]:.6f} {uv[1]:.6f}")
            obj_lines.append(f"usemtl {mat_name}")

            if write_uv:
                for i, tri in enumerate(tris):
                    a = tri[0] + 1 + v_off
                    b = tri[1] + 1 + v_off
                    c = tri[2] + 1 + v_off
                    ta = i * 3 + 1 + vt_off
                    tb = i * 3 + 2 + vt_off
                    tc = i * 3 + 3 + vt_off
                    obj_lines.append(f"f {a}/{ta} {b}/{tb} {c}/{tc}")
                vt_off += len(tuvs)
            else:
                for tri in tris:
                    a = tri[0] + 1 + v_off
                    b = tri[1] + 1 + v_off
                    c = tri[2] + 1 + v_off
                    obj_lines.append(f"f {a} {b} {c}")

            v_off += len(verts)

        if write_mesh_files:
            with open(out_obj_path, "w") as f:
                f.write("\n".join(obj_lines) + "\n")
            with open(out_mtl_path, "w") as f:
                f.write("\n".join(mtl_lines) + "\n")
            print(f"Exported: {out_obj_path}")

        if make_zip and written:
            zip_path = base + "_textures.zip"
            with zipfile.ZipFile(
                zip_path,
                "w",
                zipfile.ZIP_DEFLATED,
            ) as zf:
                for tp in written:
                    zf.write(
                        tp,
                        os.path.join(tex_subdir, os.path.basename(tp)),
                    )
            print(f"Created ZIP containing textures: {zip_path}")
