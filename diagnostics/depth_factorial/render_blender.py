"""Physically render B: fixed CAD, K, pose, light, and camera translation only.

Run INSIDE Blender, not regular Python:
  blender -b --python diagnostics/depth_factorial/render_blender.py -- \
      --plan /ABS/controlled_plan.json --output /ABS/new_render_source

Cycles depth and object-index EXR passes become strict top-left depth_m.npy
and mask.png. A ray-cast spot check verifies camera-z metres, not ray length.
No image zoom is used to generate different physical distances.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

repo=Path(__file__).resolve().parents[2]
if str(repo) not in sys.path:sys.path.insert(0,str(repo))
from diagnostics.depth_factorial.factorial_contract import (
    PLAN_SCHEMA,SOURCE_SCHEMA,digest,sha256,write_json,
    validate_plan,expected_c2w,verify_fixed_physics
)


def _require_blender():
    try:
        import bpy
        from mathutils import Matrix,Vector
        from bpy_extras.object_utils import world_to_camera_view
        return bpy,Matrix,Vector,world_to_camera_view
    except ImportError as err:
        raise RuntimeError("Run this module via Blender's bundled Python") from err


def _new_material(bpy,name,rgb,roughness=.85):
    mat=bpy.data.materials.new(name)
    mat.use_nodes=True
    node=mat.node_tree.nodes.get("Principled BSDF")
    node.inputs["Base Color"].default_value=tuple(float(v) for v in rgb)+(1.,)
    node.inputs["Roughness"].default_value=roughness
    return mat


def _load_mesh(bpy,Matrix,Vector,plan):
    target=np.asarray(plan["target_world"],dtype=float)
    o=plan["object"]
    if o["kind"]=="procedural_box":
        bpy.ops.mesh.primitive_cube_add(size=1,location=(0,0,0))
        item=bpy.context.object
        item.dimensions=tuple(o["dimensions_m"])
        bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
        objects=[item]
    else:
        path=Path(o["path"]).resolve()
        before=set(bpy.data.objects)
        suffix=path.suffix.lower()
        if suffix==".obj":
            if hasattr(bpy.ops.wm,"obj_import"):bpy.ops.wm.obj_import(filepath=str(path))
            else:bpy.ops.import_scene.obj(filepath=str(path))
        elif suffix==".ply":
            if hasattr(bpy.ops.wm,"ply_import"):bpy.ops.wm.ply_import(filepath=str(path))
            else:bpy.ops.import_mesh.ply(filepath=str(path))
        elif suffix==".stl":
            if hasattr(bpy.ops.wm,"stl_import"):bpy.ops.wm.stl_import(filepath=str(path))
            else:bpy.ops.import_mesh.stl(filepath=str(path))
        elif suffix==".glb":bpy.ops.import_scene.gltf(filepath=str(path))
        else:raise ValueError(f"Unsupported CAD mesh suffix {suffix}")
        objects=[item for item in bpy.data.objects if item not in before and item.type=="MESH"]
        if not objects:raise RuntimeError("No mesh object imported")
        factor=float(o["metres_per_mesh_unit"])
        for item in objects:
            item.scale*=factor
    bpy.context.view_layer.update()
    verts=[]
    for item in objects:
        verts.extend([np.asarray(item.matrix_world@Vector(v),dtype=float)
                      for v in item.bound_box])
    vertices=np.stack(verts)
    center=(vertices.max(0)+vertices.min(0))/2.
    delta=target-center
    for item in objects:
        item.location+=Vector(delta)
        item.pass_index=1
        mat=_new_material(bpy,"fixed_object_material",(0.63,0.32,0.21))
        item.data.materials.clear();item.data.materials.append(mat)
    bpy.context.view_layer.update()
    vertices2=[]
    for item in objects:
        vertices2.extend([np.asarray(item.matrix_world@Vector(v),dtype=float)
                          for v in item.bound_box])
    vertices2=np.stack(vertices2)
    return objects,dict(
        bounds_world_min=vertices2.min(0).tolist(),
        bounds_world_max=vertices2.max(0).tolist(),
        object_centroid_world=((vertices2.max(0)+vertices2.min(0))/2).tolist())


def _setup_scene(bpy,Matrix,Vector,plan,mesh_info):
    scene=bpy.context.scene
    scene.render.engine="CYCLES"
    scene.cycles.samples=int(plan["render"]["samples"])
    scene.cycles.use_denoising=False
    scene.render.resolution_x=int(plan["camera"]["width"])
    scene.render.resolution_y=int(plan["camera"]["height"])
    scene.render.resolution_percentage=100
    scene.render.pixel_aspect_x=scene.render.pixel_aspect_y=1.
    scene.render.image_settings.file_format="PNG"
    scene.render.image_settings.color_mode="RGB"
    scene.render.image_settings.color_depth="8"
    scene.render.film_transparent=False
    scene.render.use_border=False
    scene.render.use_crop_to_border=False
    # Stable color-management parameters across all camera distances.
    scene.view_settings.view_transform="Standard"
    scene.view_settings.look="Medium High Contrast"
    scene.view_settings.exposure=0.
    scene.view_settings.gamma=1.
    scene.world.color=(.15,.15,.15)
    if scene.world.use_nodes:
        bg=scene.world.node_tree.nodes.get("Background")
        if bg:
            bg.inputs["Color"].default_value=(.3,.32,.34,1.)
            bg.inputs["Strength"].default_value=.6
    layer=bpy.context.view_layer
    layer.use_pass_z=True
    layer.use_pass_object_index=True
    # Constant background table behind the CAD, never moves with camera.
    R=np.asarray(plan["camera"]["R_c2w"],dtype=float)
    v=R[:,2]
    target=np.asarray(plan["target_world"],dtype=float)
    max_back=float(np.dot(np.asarray(mesh_info["bounds_world_max"])-target,v))
    table_loc=target+(max(.22,max_back+.10))*v
    bpy.ops.mesh.primitive_plane_add(size=3.,location=table_loc)
    plane=bpy.context.object
    plane.name="fixed_background_plane"
    plane.rotation_euler=Vector(-v).to_track_quat("Z","Y").to_euler()
    plane.pass_index=0
    plane.data.materials.append(_new_material(bpy,"fixed_table",(0.37,.38,.39)))
    # Lights are fixed in WORLD coordinates, not head-mounted to the camera.
    for i,offset in enumerate(((-.25,.24,-.20),(.32,-.12,-.13))):
        light_data=bpy.data.lights.new(f"fixed_light_{i}","AREA")
        light=bpy.data.objects.new(f"fixed_light_{i}",light_data)
        scene.collection.objects.link(light)
        light.location=Vector(target+R@np.asarray(offset,dtype=float))
        light_data.energy=220. if i==0 else 120.
        light_data.shape="DISK";light_data.size=.8
        light.rotation_euler=(Vector(target)-light.location).to_track_quat("-Z","Y").to_euler()
    camera_data=bpy.data.cameras.new("fixed_intrinsic_camera")
    camera=bpy.data.objects.new("fixed_intrinsic_camera",camera_data)
    scene.collection.objects.link(camera)
    scene.camera=camera
    w=plan["camera"]["width"];h=plan["camera"]["height"]
    fx=plan["camera"]["fx"]
    camera_data.type="PERSP"
    camera_data.sensor_fit="HORIZONTAL"
    camera_data.sensor_width=36.
    camera_data.lens=float(fx)*camera_data.sensor_width/float(w)
    camera_data.shift_x=0;camera_data.shift_y=0
    camera_data.clip_start=.01;camera_data.clip_end=5.
    # OpenCV +X right, +Y down, +Z forward. Blender +X right,
    # +Y up, -Z forward; this is a proper 180-degree rotation about X.
    Rcv=np.asarray(plan["camera"]["R_c2w"],dtype=float)
    Rb=Rcv@np.diag([1.,-1.,-1.])
    return scene,camera,Rb


def _check_calibrated_projection(scene,camera,Rcv,center,plan,world_to_camera_view,Vector):
    w=plan["camera"]["width"];h=plan["camera"]["height"]
    fx=plan["camera"]["fx"];fy=plan["camera"]["fy"]
    cx=plan["camera"]["cx"];cy=plan["camera"]["cy"]
    xyz=np.array([[0.,0.,.65],[.04,.02,.75],[-.03,-.05,.55]],dtype=float)
    err=0.
    for point in xyz:
        P=center+Rcv@point
        actual=world_to_camera_view(scene,camera,Vector(P))
        pixel=(actual.x*w,(1.-actual.y)*h)
        expected=(fx*point[0]/point[2]+cx,fy*point[1]/point[2]+cy)
        err=max(err,float(np.max(np.abs(np.array(pixel)-expected))))
    if err>.25:
        raise RuntimeError(f"Blender camera projection and exact K differ by {err}px")
    return err


def _exr_scalar(bpy,path,h,w):
    image=bpy.data.images.load(str(path),check_existing=False)
    try:
        pix=np.asarray(image.pixels[:],dtype=np.float32)
        if pix.size==h*w:result=pix.reshape(h,w)
        elif pix.size==h*w*4:result=pix.reshape(h,w,4)[:,:,0]
        else:raise RuntimeError(f"Blender EXR pass channel count unexpected: {pix.size}")
        return result[::-1].copy()  # Blender pixels bottom-left -> PNG top-left
    finally:
        bpy.data.images.remove(image)


def _one_pass_file(folder,prefix):
    files=sorted(Path(folder).glob(f"{prefix}*.exr"))
    if len(files)!=1:
        raise RuntimeError(f"Expected one Compositor depth/index EXR: {prefix}, got {files}")
    return files[0]


def _save_mask_png(bpy,path,mask):
    h,w=mask.shape
    img=bpy.data.images.new("signed_segmentation",width=w,height=h,alpha=True)
    pixels=np.zeros((h,w,4),dtype=np.float32)
    pixels[:,:,:3]=mask[:,:,None].astype(np.float32)
    pixels[:,:,3]=1.
    img.pixels.foreach_set(pixels[::-1].ravel())
    img.filepath_raw=str(path)
    img.file_format="PNG"
    img.save()
    bpy.data.images.remove(img)


def _raycheck(bpy,Vector,depth,mask,plan,T,object_set,*,n=24):
    H,W=mask.shape
    ys,xs=np.nonzero(mask&(depth>0))
    if len(xs)<n:
        raise RuntimeError("Too few visible object pixels for depth ray contract")
    ids=np.linspace(0,len(xs)-1,n,dtype=int)
    R=np.asarray(T[:3,:3],dtype=float)
    C=np.asarray(T[:3,3],dtype=float)
    fx=plan["camera"]["fx"];fy=plan["camera"]["fy"]
    cx=plan["camera"]["cx"];cy=plan["camera"]["cy"]
    dag=bpy.context.evaluated_depsgraph_get()
    optical=[];radial=[];observed=[]
    for i in ids:
        u=int(xs[i]);v=int(ys[i])
        ray=np.array([(u+.5-cx)/fx,(v+.5-cy)/fy,1.],dtype=float)
        direction=R@(ray/np.linalg.norm(ray))
        hit,location,normal,face,obj,mat=bpy.context.scene.ray_cast(
            dag,Vector(C),Vector(direction),distance=4.5)
        hit_name=getattr(getattr(obj,"original",obj),"name",None) if obj is not None else None
        if not hit or hit_name not in object_set:
            continue
        vec=np.asarray(location,dtype=float)-C
        optical.append(float(np.dot(vec,R[:,2])))
        radial.append(float(np.linalg.norm(vec)))
        observed.append(float(depth[v,u]))
    if len(observed)<10:
        raise RuntimeError("Render object-index pass and ray geometry poorly aligned")
    obs=np.array(observed);z=np.array(optical);ran=np.array(radial)
    err_z=float(np.median(np.abs(obs-z)))
    err_r=float(np.median(np.abs(obs-ran)))
    if err_z<.008:
        kind="optical_z"
    elif err_r<.008:
        kind="euclidean_range"
    else:
        raise RuntimeError(f"Depth render pass does not match camera ray-cast in metres: z={err_z}, range={err_r}")
    return kind,len(obs),err_z,err_r


def render(plan_file,output):
    bpy,Matrix,Vector,world_to_camera_view=_require_blender()
    plan=json.loads(Path(plan_file).read_text(encoding="utf-8"))
    validate_plan(plan)
    out=Path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Render source output must be a new empty directory")
    out.mkdir(parents=True,exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.context.scene.render.film_transparent=False
    bpy.context.scene.world.use_nodes=True
    objects,mesh_info=_load_mesh(bpy,Matrix,Vector,plan)
    scene,camera,Rb=_setup_scene(bpy,Matrix,Vector,plan,mesh_info)
    object_set={x.name for x in objects}
    Rcv=np.asarray(plan["camera"]["R_c2w"],dtype=float)
    H=plan["camera"]["height"];W=plan["camera"]["width"]
    # Explicit compositor links for linear, untonemapped pass values.
    scene.use_nodes=True
    tree=scene.node_tree;tree.nodes.clear()
    render_layers=tree.nodes.new("CompositorNodeRLayers")
    nodes={}
    for name,socket in (("zpass","Depth"),("index","IndexOB")):
        node=tree.nodes.new("CompositorNodeOutputFile")
        node.format.file_format="OPEN_EXR"
        node.format.color_mode="BW"
        node.format.color_depth="32"
        node.file_slots[0].path=name+"_"
        tree.links.new(render_layers.outputs[socket],node.inputs[0])
        nodes[name]=node
    views=[]
    for i,d in enumerate(plan["distances_m"]):
        folder=out/f"distance_{i:03d}"
        folder.mkdir(parents=True,exist_ok=False)
        T=expected_c2w(plan,d)
        camera.matrix_world=Matrix(np.vstack([np.c_[Rb,T[:3,3]],[[0,0,0,1]]]).tolist())
        bpy.context.view_layer.update()
        projection_err=_check_calibrated_projection(
            scene,camera,Rcv,T[:3,3],plan,world_to_camera_view,Vector)
        for name,node in nodes.items():
            node.base_path=str(folder)
        scene.render.filepath=str(folder/"rgb.png")
        scene.frame_set(1)
        bpy.ops.render.render(write_still=True)
        zpath=_one_pass_file(folder,"zpass_")
        ipath=_one_pass_file(folder,"index_")
        raw_depth=_exr_scalar(bpy,zpath,H,W)
        index=_exr_scalar(bpy,ipath,H,W)
        mask=index>.5
        if not np.any(mask):raise RuntimeError(f"Object disappeared at distance {d}")
        depth=np.where(np.isfinite(raw_depth)&(raw_depth>0)&(raw_depth<5),
                       raw_depth,0.).astype(np.float32)
        kind,n,err_z,err_r=_raycheck(bpy,Vector,depth,mask,plan,T,object_set)
        if kind=="euclidean_range":
            yy,xx=np.indices((H,W))
            radial_factor=np.sqrt(1.+((xx+.5-plan["camera"]["cx"])/plan["camera"]["fx"])**2+
                                  ((yy+.5-plan["camera"]["cy"])/plan["camera"]["fy"])**2)
            depth=depth/radial_factor.astype(np.float32)
        depth=np.where(mask,depth,0.).astype(np.float32)
        if not np.isfinite(depth).all() or np.any(depth[mask]<=0):
            raise RuntimeError("Rendered metric camera Z invalid on object")
        source=folder/"depth_m.npy";np.save(source,depth,allow_pickle=False)
        seg=folder/"seg.png";_save_mask_png(bpy,seg,mask)
        paths={name:{"path":str(path.resolve()),"sha256":sha256(path)}
               for name,path in (("rgb",folder/"rgb.png"),
                                 ("depth",source),("seg",seg))}
        views.append(dict(distance_m=float(d),index=i,
            K=[[plan["camera"]["fx"],0.,plan["camera"]["cx"]],
               [0.,plan["camera"]["fy"],plan["camera"]["cy"]],[0.,0.,1.]],
            T_c2w=T.tolist(),files=paths,
            object_mask_pixels=int(mask.sum()),
            depth_reference_kind="rendered_object_camera_z_metres",
            projection_max_error_px=projection_err,
            raycast_checks=n,raw_pass_kind=kind,
            raycast_z_mae_m=err_z,raycast_range_mae_m=err_r))
        for tmp in (zpath,ipath):tmp.unlink()
        print(f"[Blender B] {i+1}/{len(plan['distances_m'])} dist={d}m "
              f"pixels={mask.sum()} ray_parity={err_z:.5f}",flush=True)
    result=dict(schema=SOURCE_SCHEMA,complete=True,
        plan=plan,plan_sha256=digest(plan),plan_file_sha256=sha256(plan_file),
        blender_version=bpy.app.version_string,
        mesh_info=mesh_info,views=views,
        experimental_scope="fixed mesh/material/light/camera K/rotation; physical optical-axis translation only",
        no_model_training=True,no_grasp_AP=True)
    verify_fixed_physics(result,strict_files=True)
    write_json(out/"manifest.json",result)
    return result


def main():
    args=sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args(args)
    render(a.plan,a.output)


if __name__=="__main__":
    main()
