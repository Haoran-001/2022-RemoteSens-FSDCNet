import bpy
import os
import numpy as np, math
from os import listdir
from os.path import isfile, join
import glob
import random

# create a sphere
#bpy.ops.mesh.primitive_uv_sphere_add(radius=1, enter_editmode=False, 
#    align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
#bpy.ops.transform.translate(value=(-3.15158, -5.02222, 1.71423), 
#    orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), 
#    orient_matrix_type='GLOBAL', mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', 
#    proportional_size=1, use_proportional_connected=False, 
#    use_proportional_projected=False)
#bpy.ops.object.shade_smooth()

#imported_uv = bpy.context.selected_objects[0]
#imported_uv.scale  = (0.5, 0.5, 0.5)

## paste a material for the sphere
#if len(bpy.data.materials) == 0:
#    bpy.data.materials.new('material')
#    bpy.data.materials[0].use_nodes = True
#    bpy.data.materials["material"].node_tree.nodes["Principled BSDF"].inputs[0].default_value[0] = 0
#    bpy.data.materials["material"].node_tree.nodes["Principled BSDF"].inputs[0].default_value[1] = 0
#    bpy.data.materials["material"].node_tree.nodes["Principled BSDF"].inputs[0].default_value[2] = 0
#    imported_uv.data.materials.append(bpy.data.materials[0])

# create a particle system
if len(bpy.data.particles) == 0:
    bpy.ops.object.particle_system_add()
    bpy.data.particles["ParticleSettings"].type = 'HAIR'
    bpy.data.particles["ParticleSettings"].use_advanced_hair = True
    bpy.data.particles["ParticleSettings"].count = 2000
    bpy.data.particles["ParticleSettings"].hair_length = 100
    bpy.data.particles["ParticleSettings"].emit_from = 'VERT'
    bpy.data.particles["ParticleSettings"].use_emit_random = True
    bpy.data.particles["ParticleSettings"].render_type = 'OBJECT'
    bpy.data.particles["ParticleSettings"].instance_object = imported_uv
    bpy.data.particles["ParticleSettings"].particle_size = 0.001


models = sorted(glob.glob('pcd_dir/*.pcd'))
nviews = 12
context = bpy.context
scene = bpy.context.scene
missed_models_list = []

def random_project():
    x = random.randint(0, 359)
    y = random.randint(0, 359)
    z = random.randint(0, 359)
    return (x, y, z)

for model_path in models:
    print(model_path)
    try:
        bpy.ops.import_scene.import_pcd(filepath = model_path, filter_glob = '*.pcd')
    except:
        missed_models_list.append(model_path)
        
    imported_pcd = bpy.context.selected_objects[0]
    bpy.ops.object.origin_set(type = 'ORIGIN_GEOMETRY')
    imported_pcd.location = (0, 0, 0)
    bpy.ops.view3d.camera_to_view_selected()
    
    imported_pcd.modifiers.new('ParticleSystem', type = 'PARTICLE_SYSTEM')
    bpy.data.particles[0].count = len(imported_pcd.data.vertices)
    imported_pcd.particle_systems['ParticleSystem'].settings = bpy.data.particles[0]
    
    imported_pcd.rotation_mode = 'XYZ'
    
    views = np.linspace(0, 2 * np.pi, nviews, endpoint=False)
    print (views)
    
    for i in range(nviews):
        imported_pcd.rotation_euler[2] = views[i]
        imported_pcd.rotation_euler[0] = np.pi
        # x, y, z = random_project()
        # imported_pcd.rotation_euler[0] = math.radians(float(x))
        # imported_pcd.rotation_euler[1] = math.radians(float(y))
        # imported_pcd.rotation_euler[2] = math.radians(float(z))
        filename = model_path.split("/")[-1]
        print (filename)
        bpy.ops.view3d.camera_to_view_selected()
        # context.scene.render.filepath = model_path+"_whiteshaded_v"+str(i)+".png"
        context.scene.render.filepath = model_path.rsplit('/', 1)[-2] + '/random_orth/' + model_path.rsplit('/', 1)[-1] + "_whiteshaded_v"+str(i)+".png"
        bpy.ops.render.render( write_still=True )
        
    meshes_to_remove = []
    for ob in bpy.context.selected_objects:
        meshes_to_remove.append(ob.data)
    bpy.ops.object.delete()
    bpy.data.particles.remove(bpy.data.particles[1])
    # Remove the meshes from memory too
    for mesh in meshes_to_remove:
        bpy.data.meshes.remove(mesh)
    
    imported_pcd = None
    del imported_pcd
