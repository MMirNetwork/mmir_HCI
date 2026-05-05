import open3d as o3d

dateiname = input("Bitte Dateipfad eingeben: ")

mesh_scan = o3d.io.read_triangle_mesh(dateiname)
pcd_scan = mesh_scan.sample_points_uniformly(number_of_points=1000000)

o3d.visualization.draw_geometries_with_editing([pcd_scan], window_name="Cropping Tool")