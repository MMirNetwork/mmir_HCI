import os, zipfile, tempfile
import numpy as np
import open3d as o3d


def load_mesh(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".obj":
        return o3d.io.read_triangle_mesh(path, enable_post_processing=True)
    elif ext in (".glb", ".gltf"):
        import trimesh
        raw = trimesh.load(path)
        tm = trimesh.util.concatenate(list(raw.geometry.values())) if hasattr(raw, "geometry") else raw
        m = o3d.geometry.TriangleMesh()
        m.vertices  = o3d.utility.Vector3dVector(np.array(tm.vertices,  dtype=np.float64))
        m.triangles = o3d.utility.Vector3iVector(np.array(tm.faces,     dtype=np.int32))
        if hasattr(tm.visual, "uv") and tm.visual.uv is not None:
            idx = np.array(tm.faces, dtype=np.int32)
            m.triangle_uvs = o3d.utility.Vector2dVector(np.array(tm.visual.uv, dtype=np.float64)[idx.ravel()])
            mat = tm.visual.material
            if hasattr(mat, "baseColorTexture") and mat.baseColorTexture is not None:
                m.textures = [o3d.geometry.Image(np.array(mat.baseColorTexture.convert("RGB"), dtype=np.uint8))]
                m.triangle_material_ids = o3d.utility.IntVector(np.zeros(len(tm.faces), dtype=np.int32))
                m.vertex_colors = o3d.utility.Vector3dVector(np.array(tm.visual.to_color().vertex_colors, dtype=np.float64)[:, :3] / 255.0)
        m.compute_vertex_normals()
        return m
    elif ext == ".usdz":
        from pxr import Usd, UsdGeom
        from PIL import Image
        stage = Usd.Stage.Open(path)
        verts_all, faces_all, uvs_all, off = [], [], [], 0
        for prim in stage.Traverse():
            if prim.GetTypeName() != "Mesh": continue
            um = UsdGeom.Mesh(prim)
            pts  = np.array(um.GetPointsAttr().Get(), dtype=np.float64)
            fidx = np.array(um.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
            verts_all.append(pts); faces_all.append(fidx.reshape(-1, 3) + off); off += len(pts)
            st = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
            if st and st.IsDefined():
                uv_vals = np.array(st.Get(), dtype=np.float64)
                uv_idx  = st.GetIndices()
                uvs_all.append((uv_vals[np.array(uv_idx, dtype=np.int32)] if uv_idx is not None and len(uv_idx) else uv_vals).reshape(-1, 2))
        m = o3d.geometry.TriangleMesh()
        all_v = np.concatenate(verts_all); all_f = np.concatenate(faces_all)
        m.vertices  = o3d.utility.Vector3dVector(all_v)
        m.triangles = o3d.utility.Vector3iVector(all_f)
        if uvs_all:
            all_uv = np.concatenate(uvs_all)
            if len(all_uv) == len(all_f) * 3:
                m.triangle_uvs = o3d.utility.Vector2dVector(all_uv)
                with zipfile.ZipFile(path) as z:
                    cands = [n for n in z.namelist() if n.lower().endswith(("_tex0.png", "_tex0.jpg"))] \
                         or [n for n in z.namelist() if n.lower().endswith(".png")]
                    if cands:
                        with tempfile.TemporaryDirectory() as tmp:
                            z.extract(cands[0], tmp)
                            img = Image.open(os.path.join(tmp, cands[0])).convert("RGB")
                            tex = np.array(img, dtype=np.float32) / 255.0
                            m.textures = [o3d.geometry.Image(np.array(img, dtype=np.uint8))]
                            m.triangle_material_ids = o3d.utility.IntVector(np.zeros(len(all_f), dtype=np.int32))
                            th, tw = tex.shape[:2]
                            px = np.clip(all_uv[:, 0], 0, .999) * (tw - 1)
                            py = np.clip(1 - all_uv[:, 1], 0, .999) * (th - 1)
                            corner_colors = tex[py.astype(np.int32), px.astype(np.int32)]
                            vcol = np.zeros((len(all_v), 3), dtype=np.float32)
                            vcnt = np.zeros(len(all_v), dtype=np.int32)
                            for ci, vi in enumerate(all_f.ravel()):
                                vcol[vi] += corner_colors[ci]; vcnt[vi] += 1
                            mask = vcnt > 0; vcol[mask] /= vcnt[mask, None]
                            m.vertex_colors = o3d.utility.Vector3dVector(vcol.astype(np.float64))
        m.compute_vertex_normals()
        return m
    else:
        raise ValueError(f"Unsupported format: {ext}")


dateiname = input("Bitte Dateipfad eingeben: ")

mesh_scan = load_mesh(dateiname)
pcd_scan = mesh_scan.sample_points_uniformly(number_of_points=1000000)

o3d.visualization.draw_geometries_with_editing([pcd_scan], window_name="Cropping Tool")