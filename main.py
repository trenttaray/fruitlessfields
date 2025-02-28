# In main.py
# Specify dependencies
# /// script
# dependencies = [
#    "panda3d",
#    "numpy",
# ]
# ///
#pip install numpy==2.0.2
# Import standard modules
import os
import sys
import time
import random
import math
import json
import copy
import numpy as np
import ast
# Import Panda3D modules
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from direct.gui.OnscreenImage import OnscreenImage
from direct.gui.DirectGui import DirectButton, DirectSlider, OnscreenText, DirectFrame, DirectWaitBar
from panda3d.core import (
    Geom, GeomTriangles, GeomVertexFormat, GeomVertexData, GeomVertexWriter, 
    TextureStage, GeomNode, PointLight, DirectionalLight, Spotlight, 
    AmbientLight, Vec4, GeomVertexReader, LVecBase3f, CollisionTraverser, 
    CollisionNode, CollisionPolygon, CollisionHandlerQueue, CollisionRay, 
    BitMask32, Point3, Vec3, NodePath, VirtualFileSystem, Multifile, 
    getModelPath, PNMImage, Texture, Filename, TransparencyAttrib, Shader, 
    OmniBoundingVolume, GeomVertexArrayFormat, BoundingBox, WindowProperties, 
    loadPrcFileData, LPoint3f, BoundingSphere, LVecBase4f, LVecBase4,AudioManager, AudioSound,Mat4,LVector3f,CollisionBox, CompassEffect,CullBinManager,DepthWriteAttrib,PerspectiveLens,CollisionSphere

)
from panda3d.physics import ForceNode, LinearVectorForce
from direct.stdpy import threading
from direct.stdpy.threading2 import Thread
from direct.particles.ParticleEffect import ParticleEffect  # Add this line
from panda3d.core import TransformState
# Local import
from characterController.PlayerController import PlayerController


from panda3d.core import PStatCollector
from panda3d.core import Notify
from direct.showbase.Audio3DManager import Audio3DManager


# Define a decorator to profile specific functions
def pstat_collector(name):
    collector = PStatCollector(name)
    def decorator(func):
        def wrapper(*args, **kwargs):
            collector.start()
            result = func(*args, **kwargs)
            collector.stop()
            return result
        return wrapper
    return decorator

# Texture size
texture_size = 512
# Area size
area_size = 486

# Scaling factor
scale_factor = texture_size / area_size

# vertex shader instance
v_shader = '''#version 330

struct p3d_DirectionalLightParameters {
    vec4 color;
    vec3 direction;
    sampler2DShadow shadowMap;
    mat4 shadowViewMatrix;
};

uniform p3d_DirectionalLightParameters my_directional_light;
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat3 p3d_NormalMatrix;
uniform mat4 p3d_ModelViewMatrix;

in vec4 p3d_Vertex;
in vec3 p3d_Normal;
in vec2 p3d_MultiTexCoord0;
in vec4 offset;
in vec4 rotation; // Heading, Pitch, Roll
in vec4 scale;

out vec2 uv;
out vec4 shadow_uv;
out vec3 normal;
out vec4 fragPos;

mat4 rotationMatrixX(float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return mat4(
        1.0, 0.0, 0.0, 0.0,
        0.0, c, -s, 0.0,
        0.0, s, c, 0.0,
        0.0, 0.0, 0.0, 1.0
    );
}

mat4 rotationMatrixY(float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return mat4(
        c, 0.0, s, 0.0,
        0.0, 1.0, 0.0, 0.0,
        -s, 0.0, c, 0.0,
        0.0, 0.0, 0.0, 1.0
    );
}

mat4 rotationMatrixZ(float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return mat4(
        c, -s, 0.0, 0.0,
        s, c, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    );
}

void main() {
    vec4 vertexPosition = p3d_Vertex;
    vec3 transformedNormal = p3d_Normal;

    // Apply uniform scale
    vertexPosition *= scale;

    // Convert degrees to radians
    float angleH = radians(rotation.x); // Heading (yaw)
    float angleP = radians(rotation.y); // Pitch
    float angleR = radians(rotation.z); // Roll

    mat4 rotationMatrix =  rotationMatrixX(angleP) * rotationMatrixY(angleR) * rotationMatrixZ(angleH);


    vertexPosition = rotationMatrix * vertexPosition;

    transformedNormal = normalize(mat3(rotationMatrix) * p3d_Normal);

    // Apply offset
    vertexPosition += offset;

    // Position
    gl_Position = p3d_ModelViewProjectionMatrix * vertexPosition;

    // Normal
    normal = p3d_NormalMatrix * transformedNormal;

    // UV
    uv = p3d_MultiTexCoord0;

    // Shadows
    shadow_uv = my_directional_light.shadowViewMatrix * (p3d_ModelViewMatrix * vertexPosition);

    // Frag position
    fragPos = p3d_ModelViewMatrix * vertexPosition;
}'''
# fragment shader instance
f_shader = '''#version 330

struct p3d_DirectionalLightParameters {
    vec4 color;
    vec3 direction;
    sampler2DShadow shadowMap;
    mat4 shadowViewMatrix;
};

struct p3d_PointLightParameters {
    vec4 color;
    vec3 position;
    samplerCube shadowMap;
    vec3 attenuation;
};

struct p3d_SpotLightParameters {
    vec4 color;
    vec3 position;
    vec3 spotDirection;
    sampler2DShadow shadowMap;
    mat4 shadowViewMatrix;
    vec3 attenuation;
};

const int MAX_POINT_LIGHTS = 4;
const int MAX_SPOT_LIGHTS = 4;

uniform p3d_DirectionalLightParameters my_directional_light;
uniform p3d_PointLightParameters point_lights[MAX_POINT_LIGHTS];
uniform p3d_SpotLightParameters spot_lights[MAX_SPOT_LIGHTS];

uniform sampler2D p3d_Texture0;
uniform vec3 camera_pos;
uniform float shadow_blur;
uniform vec4 ambientLightColor;
uniform vec4 fogColor;
uniform float fogStart;
uniform float fogEnd;
uniform vec3 player_pos;
uniform bool enable_transparency;
uniform vec4 horizonColorb;

uniform int num_point_lights;
uniform int num_spot_lights;

in vec2 uv;
in vec4 shadow_uv;
in vec3 normal;
in vec4 fragPos;

out vec4 color;

float textureProjSoft(sampler2DShadow tex, vec4 uv, float bias, float blur) {
    float result = textureProj(tex, uv, bias);
    result += textureProj(tex, vec4(uv.xy + vec2(-0.326212, -0.405805) * blur, uv.z - bias, uv.w));
    result += textureProj(tex, vec4(uv.xy + vec2(-0.840144, -0.073580) * blur, uv.z - bias, uv.w));
    result += textureProj(tex, vec4(uv.xy + vec2(-0.695914, 0.457137) * blur, uv.z - bias, uv.w));
    result += textureProj(tex, vec4(uv.xy + vec2(-0.203345, 0.620716) * blur, uv.z - bias, uv.w));
    return result / 5.0; // Reduced number of samples
}

float calculatePointLightShadow(vec3 fragPos, vec3 lightPos, samplerCube shadowMap) {
    vec3 lightToFrag = fragPos - lightPos;
    float currentDepth = length(lightToFrag);
    float shadow = texture(shadowMap, lightToFrag).r;
    float bias = 0.05; // Adjust bias as needed
    return currentDepth - bias > shadow ? 0.5 : 1.0;
}

void main() {
    // Base color
    vec3 ambient = ambientLightColor.rgb;
    vec4 tex = texture(p3d_Texture0, uv);

    // Ensure the normal is normalized
    vec3 normalizedNormal = normalize(normal);

    // Calculate directional light contribution
    vec3 dirLight = my_directional_light.color.rgb * max(dot(normalizedNormal, my_directional_light.direction), 0.0);
    float dirLightShadow = textureProjSoft(my_directional_light.shadowMap, shadow_uv, 0.0001, shadow_blur);
    dirLightShadow = 0.5 + dirLightShadow * 0.5;
    dirLight *= dirLightShadow;

    // Calculate point light contributions with attenuation
    vec3 totalPointLight = vec3(0.0);
    for (int i = 0; i < num_point_lights; i++) {
        vec3 lightDir = point_lights[i].position - fragPos.xyz;
        float distance = length(lightDir);
        vec3 attenuationFactors = point_lights[i].attenuation; // Fetch attenuation from struct
        float attenuation = 1.0 / (attenuationFactors.x + attenuationFactors.y * distance + attenuationFactors.z * (distance * distance));
        vec3 pointLight = point_lights[i].color.rgb * max(dot(normalizedNormal, normalize(lightDir)), 0.0);
        pointLight *= attenuation;
        pointLight *= calculatePointLightShadow(fragPos.xyz, point_lights[i].position, point_lights[i].shadowMap);
        totalPointLight += pointLight;
    }

    // Calculate spotlight contributions
    vec3 totalSpotLight = vec3(0.0);
    for (int i = 0; i < num_spot_lights; i++) {
        vec3 spotDirection = normalize(spot_lights[i].spotDirection);
        vec3 lightDir = spot_lights[i].position - fragPos.xyz;
        float distance = length(lightDir);
        vec3 attenuationFactors = spot_lights[i].attenuation;
        float attenuation = 1.0 / (attenuationFactors.x + attenuationFactors.y * distance + attenuationFactors.z * (distance * distance)); // Use attenuation from struct
        vec3 spotLight = spot_lights[i].color.rgb * max(dot(normalizedNormal, -spotDirection), 0.0);
        float theta = dot(normalize(fragPos.xyz - spot_lights[i].position), spotDirection);
        float intensity = max(pow(theta, 10.0), 0.0); // Adjust the exponent to control the spotlight focus
        spotLight *= intensity * attenuation;
        totalSpotLight += spotLight;
    }

    // Combine all lighting
    vec3 finalLight = dirLight + ambient + totalPointLight + totalSpotLight;

    // Precompute fog factor
    float heightFogFactor = clamp((fogEnd - length(fragPos.xyz.y)) / (fogEnd - fogStart), 0.0, 1.0);
    float depthFogFactor = clamp((fogEnd - length(fragPos.xyz)) / (fogEnd - fogStart), 0.0, 1.0);
    float fogFactor = min(heightFogFactor, depthFogFactor);

    // Blend fog color with skybox color at the horizon
    vec4 horizonColor = mix(fogColor, horizonColorb, 0.5);
    vec4 foggedColor = mix(horizonColor, vec4(tex.rgb * finalLight, tex.a), fogFactor);

    // Calculate distance from player position
    float distance = length(fragPos.xyz - player_pos);

    // Define a threshold for alpha
    float alphaThreshold = 0.5;

    // Adjust alpha based on distance with a hard cut-off
    float alpha = enable_transparency ? (distance < 24.0 ? 0.0 : (tex.a > alphaThreshold ? 1.0 : 0.0)) : (tex.a > alphaThreshold ? 1.0 : 0.0);

    // Apply the alpha to the fogged color
    color = vec4(foggedColor.rgb, alpha);
}'''

shader01v = '''#version 330

struct p3d_DirectionalLightParameters {
    vec4 color;
    vec3 direction;
    sampler2DShadow shadowMap;
    mat4 shadowViewMatrix;
};

uniform p3d_DirectionalLightParameters my_directional_light;
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat3 p3d_NormalMatrix;
uniform mat4 p3d_ModelViewMatrix;

in vec4 p3d_Vertex;
in vec3 p3d_Normal;
in vec2 p3d_MultiTexCoord0;

out vec2 uv;
out vec4 shadow_uv;
out vec3 normal;
out vec4 fragPos;

void main() {
    vec4 vertexPosition = p3d_Vertex;
    vec3 transformedNormal = p3d_Normal;

    // Position
    gl_Position = p3d_ModelViewProjectionMatrix * vertexPosition;

    // Normal
    normal = p3d_NormalMatrix * transformedNormal;

    // UV
    uv = p3d_MultiTexCoord0;

    // Shadows
    shadow_uv = my_directional_light.shadowViewMatrix * (p3d_ModelViewMatrix * vertexPosition);

    // Frag position
    fragPos = p3d_ModelViewMatrix * vertexPosition;
}'''

shader01f = '''#version 330

struct p3d_DirectionalLightParameters {
    vec4 color;
    vec3 direction;
    sampler2DShadow shadowMap;
    mat4 shadowViewMatrix;
};

struct p3d_PointLightParameters {
    vec4 color;
    vec3 position;
    samplerCube shadowMap;
    vec3 attenuation;
};

struct p3d_SpotLightParameters {
    vec4 color;
    vec3 position;
    vec3 spotDirection;
    sampler2DShadow shadowMap;
    mat4 shadowViewMatrix;
    vec3 attenuation;
};

const int MAX_POINT_LIGHTS = 4;
const int MAX_SPOT_LIGHTS = 4;

uniform p3d_DirectionalLightParameters my_directional_light;
uniform p3d_PointLightParameters point_lights[MAX_POINT_LIGHTS];
uniform p3d_SpotLightParameters spot_lights[MAX_SPOT_LIGHTS];

uniform int num_point_lights;
uniform int num_spot_lights;

uniform sampler2D p3d_Texture0;
uniform vec3 camera_pos;
uniform float shadow_blur;
uniform vec4 ambientLightColor;
uniform vec4 fogColor;
uniform float fogStart;
uniform float fogEnd;
uniform vec3 player_pos;
uniform bool enable_transparency;
uniform vec4 horizonColorb;
uniform vec4 modelColor;

in vec2 uv;
in vec4 shadow_uv;
in vec3 normal;
in vec4 fragPos;

out vec4 color;

float textureProjSoft(sampler2DShadow tex, vec4 uv, float bias, float blur) {
    float result = textureProj(tex, uv, bias);
    result += textureProj(tex, vec4(uv.xy + vec2(-0.326212, -0.405805) * blur, uv.z - bias, uv.w));
    result += textureProj(tex, vec4(uv.xy + vec2(-0.840144, -0.073580) * blur, uv.z - bias, uv.w));
    result += textureProj(tex, vec4(uv.xy + vec2(-0.695914, 0.457137) * blur, uv.z - bias, uv.w));
    result += textureProj(tex, vec4(uv.xy + vec2(-0.203345, 0.620716) * blur, uv.z - bias, uv.w));
    return result / 5.0; // Reduced number of samples
}

float calculatePointLightShadow(vec3 fragPos, vec3 lightPos, samplerCube shadowMap) {
    vec3 lightToFrag = fragPos - lightPos;
    float currentDepth = length(lightToFrag);
    float shadow = texture(shadowMap, lightToFrag).r;
    float bias = 0.05; // Adjust bias as needed
    return currentDepth - bias > shadow ? 0.5 : 1.0;
}

void main() {
    // Base color
    vec3 ambient = ambientLightColor.rgb;
    vec4 tex = texture(p3d_Texture0, uv);

    // Ensure the normal is normalized
    vec3 normalizedNormal = normalize(normal);

    // Calculate directional light contribution
    vec3 dirLight = my_directional_light.color.rgb * max(dot(normalizedNormal, my_directional_light.direction), 0.0);
    float dirLightShadow = textureProjSoft(my_directional_light.shadowMap, shadow_uv, 0.0001, shadow_blur);
    dirLightShadow = 0.5 + dirLightShadow * 0.5;
    dirLight *= dirLightShadow;

    // Calculate point light contributions with attenuation
    vec3 totalPointLight = vec3(0.0);
    for (int i = 0; i < num_point_lights; i++) {
        vec3 lightDir = point_lights[i].position - fragPos.xyz;
        float distance = length(lightDir);
        vec3 attenuationFactors = point_lights[i].attenuation; // Fetch attenuation from struct
        float attenuation = 1.0 / (attenuationFactors.x + attenuationFactors.y * distance + attenuationFactors.z * (distance * distance));
        vec3 pointLight = point_lights[i].color.rgb * max(dot(normalizedNormal, normalize(lightDir)), 0.0);
        pointLight *= attenuation;
        pointLight *= calculatePointLightShadow(fragPos.xyz, point_lights[i].position, point_lights[i].shadowMap);
        totalPointLight += pointLight;
    }

    // Calculate spotlight contributions
    vec3 totalSpotLight = vec3(0.0);
    for (int i = 0; i < num_spot_lights; i++) {
        vec3 spotDirection = normalize(spot_lights[i].spotDirection);
        vec3 lightDir = spot_lights[i].position - fragPos.xyz;
        float distance = length(lightDir);
        vec3 attenuationFactors = spot_lights[i].attenuation;
        float attenuation = 1.0 / (attenuationFactors.x + attenuationFactors.y * distance + attenuationFactors.z * (distance * distance)); // Use attenuation from struct
        vec3 spotLight = spot_lights[i].color.rgb * max(dot(normalizedNormal, -spotDirection), 0.0);
        float theta = dot(normalize(fragPos.xyz - spot_lights[i].position), spotDirection);
        float intensity = max(pow(theta, 10.0), 0.0); // Adjust the exponent to control the spotlight focus
        spotLight *= intensity * attenuation;
        totalSpotLight += spotLight;
    }

    // Combine all lighting
    vec3 finalLight = dirLight + ambient + totalPointLight + totalSpotLight;

    // Precompute fog factor
    float heightFogFactor = clamp((fogEnd - length(fragPos.xyz.y)) / (fogEnd - fogStart), 0.0, 1.0);
    float depthFogFactor = clamp((fogEnd - length(fragPos.xyz)) / (fogEnd - fogStart), 0.0, 1.0);
    float fogFactor = min(heightFogFactor, depthFogFactor);

    // Blend fog color with skybox color at the horizon
    vec4 horizonColor = mix(fogColor, horizonColorb, 0.5); // Light sky blue color for example
    vec4 foggedColor = mix(horizonColor, vec4(tex.rgb * finalLight, tex.a), fogFactor);

    // Calculate distance from player position
    float distance = length(fragPos.xyz - player_pos);

    // Define a threshold for alpha
    float alphaThreshold = 0.5;

    // Adjust alpha based on distance with a hard cut-off
    float alpha = enable_transparency ? (distance < 24.0 ? 0.0 : (tex.a > alphaThreshold ? 1.0 : 0.0)) : (tex.a > alphaThreshold ? 1.0 : 0.0);

    // Apply the alpha to the fogged color
    color = vec4(foggedColor.rgb * modelColor.rgb, modelColor.a * alpha);
}'''


loadPrcFileData("","""
cursor-hidden 1
window-title FruitlessFields
gl-debug true

support-threads #t
fullscreen #f
#win-size 1920 1080
win-size 1080 720
#win-size 840 720
#side-by-side-stereo 1
occlusion-culling true              
#want-pstats 1
                
sync-video #f 
clock-mode limited
clock-frame-rate 60

collide-mask 0 
auto-flatten 1

textures-auto-power-2 1 
textures-srgb on 
auto-choose-mipmaps true

glsl-preprocess #t 
glsl-varying-limit 32
""")

class MyApp(ShowBase):
    def __init__(self):
        self.storework=[]
        self.activatecont=1
        self.activa=1
        self.trackamount={}
        self.instance_groups={}
        ShowBase.__init__(self)
        # Set up the window properties
        props = WindowProperties()
        props.setIconFilename("data/icon.ico")
        self.win.requestProperties(props)
        self.taskMgr.setupTaskChain('terrain_chain', numThreads=1, tickClock=False, frameSync = False)
        self.taskMgr.setupTaskChain('clearer', numThreads=1, tickClock=False)
        #transparent stuff
        # Create bins
        CullBinManager.get_global_ptr().add_bin('opaque', CullBinManager.BT_fixed, 30)
        CullBinManager.get_global_ptr().add_bin('transparent', CullBinManager.BT_fixed, 40)

        #shader
        self.shader01 = Shader.make(Shader.SL_GLSL,shader01v, shader01f)#non instanced
        # self.shader02 = Shader.make(Shader.SL_GLSL,shader01v, shader02f)

        self.shadershadows = Shader.make(Shader.SL_GLSL,v_shader, f_shader)
        #shader settings
        self.render.set_shader_input('shadow_blur',0.0005)
        self.render.set_shader_input('player_pos',(0,0,0))
        self.render.set_shader_input('enable_transparency',False)

        self.render.setShaderInput("modelColor", Vec4(1, 1, 1, 1))


        self.render.setShaderInput("fogColor", (0.5, 0.5, 0.5, 1.0)) # Set the fog color
        self.render.setShaderInput("fogStart", 300.0) # Set the fog start distance
        self.render.setShaderInput("fogEnd", 410.0) # Set the fog end distance

        # Assuming 'camera' is your camera NodePath
        camera = base.cam
        lens = camera.node().getLens()
        lens.setNear(1)
        lens.setFar(700.0)

        # Load the sound file
        self.audio_mgr = AudioManager.createAudioManager()
        # self.audio3d = Audio3DManager(self.audio_mgr, base.camera)
        
        # Create a master volume control
        self.master_volume = 1.0

        # Load and configure sounds
        self.land_sound = self.audio_mgr.getSound("data/audio/stepdirt_1.wav")
        self.land_sound.setVolume(0.003 * self.master_volume)

        self.walk_sound = self.audio_mgr.getSound("data/audio/mud02.wav")
        self.walk_sound.setLoop(True)  # Enable looping
        self.walk_sound.set3dAttributes(0, -15, 0, 0, 0, 0)
        self.walk_sound.setVolume(0.005 * self.master_volume)  # Set the volume (0.0 to 1.0)
        self.walk_sound.set3dMinDistance(10)  # Start attenuation at 10 units 
        self.walk_sound.set3dMaxDistance(50)  # Max distance at which the sound is audible

        self.run_sound = self.audio_mgr.getSound("data/audio/mud02.wav")
        
        self.run_sound.setLoop(False)  # Enable looping
        self.run_sound.set3dAttributes(0, -15, 0, 0, 0, 0)
        self.run_sound.setVolume(0.009 * self.master_volume)  # Set the volume (0.0 to 1.0)
        self.run_sound.set3dMinDistance(10)  # Start attenuation at 10 units
        self.run_sound.set3dMaxDistance(50) # Max distance at which the sound is audible
        # self.audio3d.attachSoundToObject(self.run_sound, base.camera)#causes camera problems
        self.dig_sound = self.audio_mgr.getSound("data/audio/stepdirt_7.wav")
        self.dig_sound.setVolume(0.009 * self.master_volume)

        self.chop_sound = self.audio_mgr.getSound("data/audio/Hand_Clap_01.wav")
        self.chop_sound.setVolume(0.009 * self.master_volume)
        # Example usage: Change master volume to 0.5
        self.set_master_volume(3.5)

        # Attach event listener for walking sound
        self.accept("player-play-walk-sfx", self.play_walk_sound)
        self.accept("player-stop-walk-sfx", self.stop_walk_sound)
        self.accept("player-play-run-sfx", self.play_run_sound)
        self.accept("player-play-sprint-sfx", self.play_sprint_sound)
        # self.accept("player-play-jump-sfx", self.play_walk_sound)
        self.accept("player-play-land-sfx", self.play_land_sound)
        # self.accept("player-play-fall-sfx", self.play_walk_sound)
        # self.accept("player-set-playrate-walk-sfx", self.play_walk_sound)
        

        #dev settings
        # self.setFrameRateMeter(True)
        self.render.setShaderAuto()
        self.disableMouse( )
        #buttons group01
        self.accept("escape", self.togglePause)
        self.accept("f1", self.hidePause)

        self.accept("1", self.key1)
        self.accept("2", self.key2)
        self.accept("3", self.key3)
        self.accept("4", self.key4)
        self.accept("c", self.keyC)
        self.accept("h", self.keypointhide)
        self.accept("f", self.keynormals)
        self.accept("f2", self.screenshot)
        
        
        # self.accept("G", self.keyG)
        # self.key1()
        self.key1tog = False
        self.key2tog = False
        self.key3tog = False
        self.key4tog = False
        self.keyCtog = False
        self.keynormalstog=False
        # self.accept("window-event", self.adjust_window_size)
        self.accept("window-event", self.on_window_event)
        # self.accept("close_request", self.on_close_request)
        base.win.setCloseRequestEvent('exit_stage_right')
        self.accept('exit_stage_right',self.on_close_request)
        current_wp = self.win.getProperties()
        
        # Get the current window size
        current_width = current_wp.getXSize()
        current_height = current_wp.getYSize()
        self.gui_root = NodePath('gui_root') 
        self.gui_root.reparentTo(base.aspect2d)
        # self.pause_frame = DirectFrame(scale=0.1,frameColor=(0, 0, 0, 255), frameSize=(-4, 4, -8, 2))
        self.pause_frame = DirectFrame( scale=(120.0 / current_width, 1, 80.0 / current_height), frameColor=(0, 0, 0, 255), frameSize=(-4, 4, -8, 2) )

        self.interact_frame = DirectFrame( scale=(120.0 / current_width, 1, 80.0 / current_height), frameColor=(0, 0, 0, 255), frameSize=(-4, 4, -4, -1) )
        self.interact_frame.reparentTo(self.gui_root) # Parent the frame to aspect2d
        self.interact_frame.setPos(0, 0, -0.5)

        self.onscreengui_frame = DirectFrame( scale=(120.0 / current_width, 1, 80.0 / current_height), frameColor=(0, 0, 0, 0), frameSize=(-4, 4, -4, -1) )
        self.onscreengui_frame.reparentTo(self.gui_root) # Parent the frame to aspect2d
        self.holding = OnscreenText(text='0', frame=None, pos=(0, 0), scale=0.7, fg=(1, 1, 1, 1))
        self.holding['text']=''
        self.holding.setPos(0, -2.5)
        self.holding.reparentTo(self.onscreengui_frame)

        # Add crosshair
        self.crosshair = OnscreenImage(image='data/level/crosshair.png', pos=(0, 0, 0), scale=0.6)
        self.crosshair.setTransparency(TransparencyAttrib.MAlpha)
        self.crosshair.setColor(1, 1, 1, 0.7) # Set color with alpha value

        texture = self.crosshair.getTexture()
        texture.setMinfilter(Texture.FTNearest)
        texture.setMagfilter(Texture.FTNearest)
        
        self.crosshair.reparentTo(self.onscreengui_frame)
 
        self.up_button = DirectButton(image=("data/level/arrow.png"), command=self.upfunc, frameSize=(-0.4, 0.4, -0.4, 0.4), text_scale=0.5,image_scale=(0.3, 0.3, 0.3),image_hpr=(0, 0, 0), sortOrder=1)
        self.up_button.setPos(2.5, 0, -2.5)
        self.up_button.reparentTo(self.interact_frame)
        self.down_button = DirectButton(image=("data/level/arrow.png"), command=self.downfunc, frameSize=(-0.4, 0.4, -0.4, 0.4), text_scale=0.5,image_scale=(0.3, 0.3, 0.3),image_hpr=(0, 0, 180), sortOrder=1)
        self.down_button.setPos(2.5, 0, -3.5)
        self.down_button.reparentTo(self.interact_frame)
        self.right_button = DirectButton(image=("data/level/arrow.png"), command=self.rightfunc, frameSize=(-0.4, 0.4, -0.4, 0.4), text_scale=0.5,image_scale=(0.3, 0.3, 0.3),image_hpr=(0, 0, 90), sortOrder=1)
        self.right_button.setPos(3.5, 0, -3.5)
        self.right_button.reparentTo(self.interact_frame)
        self.left_button = DirectButton(image=("data/level/arrow.png"), command=self.leftfunc, frameSize=(-0.4, 0.4, -0.4, 0.4), text_scale=0.5,image_scale=(0.3, 0.3, 0.3),image_hpr=(0, 0, -90), sortOrder=1)
        self.left_button.setPos(1.5, 0, -3.5)
        self.left_button.reparentTo(self.interact_frame)
        self.zup_button = DirectButton(text='u', command=self.zupfunc, frameSize=(-0.4, 0.4, -0.4, 0.4), text_scale=0.5, sortOrder=1)
        self.zup_button.setPos(3.5, 0, -2.5)
        self.zup_button.reparentTo(self.interact_frame)
        self.zdown_button = DirectButton(text='d', command=self.zdownfunc, frameSize=(-0.4, 0.4, -0.4, 0.4), text_scale=0.5, sortOrder=1)
        self.zdown_button.setPos(1.5, 0, -2.5)
        self.zdown_button.reparentTo(self.interact_frame)

        self.incrementp_button = DirectButton(text='1', command=self.incremntupfunc, frameSize=(-0.4, 0.4, -0.4, 0.4), text_scale=0.5, sortOrder=1)
        self.incrementp_button.setPos(3.5, 0, -1.5)
        self.incrementp_button.reparentTo(self.interact_frame)
        self.incrementn_button = DirectButton(text='0', command=self.incremntdownfunc, frameSize=(-0.4, 0.4, -0.4, 0.4), text_scale=0.5, sortOrder=1)
        self.incrementn_button.setPos(2.5, 0, -1.5)
        self.incrementn_button.reparentTo(self.interact_frame)

        self.place_button = DirectButton(text='p', command=self.placefunc, frameSize=(-0.4, 0.4, -0.4, 0.4), text_scale=0.5, sortOrder=1)
        self.place_button.setPos(-2.5, 0, -2.5)
        self.place_button.reparentTo(self.interact_frame)
        self.resetrot_button = DirectButton(text='r', command=self.resetrotfunc, frameSize=(-0.4, 0.4, -0.4, 0.4), text_scale=0.5, sortOrder=1)
        self.resetrot_button.setPos(-2.5, 0, -3.5)
        self.resetrot_button.reparentTo(self.interact_frame)

        self.rotforward_button = DirectButton(image=("data/level/arrow.png"), command=self.rotforwardfunc, frameSize=(-0.4, 0.4, -0.4, 0.4), text_scale=0.5,image_scale=(0.3, 0.3, 0.3),image_hpr=(0, 0, 0), sortOrder=1)
        self.rotforward_button.setPos(0.5, 0, -2.5)
        self.rotforward_button.reparentTo(self.interact_frame)
        self.rotdownward_button = DirectButton(image=("data/level/arrow.png"), command=self.rotdownwardfunc, frameSize=(-0.4, 0.4, -0.4, 0.4), text_scale=0.5,image_scale=(0.3, 0.3, 0.3),image_hpr=(0, 0, 180), sortOrder=1)
        self.rotdownward_button.setPos(0.5, 0, -3.5)
        self.rotdownward_button.reparentTo(self.interact_frame)
        self.rotleft_button = DirectButton(image=("data/level/arrow.png"), command=self.rotleftfunc, frameSize=(-0.4, 0.4, -0.4, 0.4), text_scale=0.5,image_scale=(0.3, 0.3, 0.3),image_hpr=(0, 0, -90), sortOrder=1)
        self.rotleft_button.setPos(-1.5, 0, -2.5)
        self.rotleft_button.reparentTo(self.interact_frame)
        self.rotright_button = DirectButton(image=("data/level/arrow.png"), command=self.rotrightfunc, frameSize=(-0.4, 0.4, -0.4, 0.4), text_scale=0.5,image_scale=(0.3, 0.3, 0.3),image_hpr=(0, 0, 90), sortOrder=1)
        self.rotright_button.setPos(-0.5, 0, -2.5)
        self.rotright_button.reparentTo(self.interact_frame)
        self.rotleftaround_button = DirectButton(image=("data/level/arrow2.png"), command=self.rotheadingleftfunc, frameSize=(-0.4, 0.4, -0.4, 0.4), text_scale=0.5,image_scale=(0.3, 0.3, 0.3),image_hpr=(0, 0, -90), sortOrder=1)
        self.rotleftaround_button.setPos(-1.5, 0, -3.5)
        self.rotleftaround_button.reparentTo(self.interact_frame)
        self.rotrightaround_button = DirectButton(image=("data/level/arrow2.png"), command=self.rotheadingrightfunc, frameSize=(-0.4, 0.4, -0.4, 0.4), text_scale=0.5,image_scale=(0.3, 0.3, 0.3),image_hpr=(0, 0, 90), sortOrder=1)
        self.rotrightaround_button.setPos(-0.5, 0, -3.5)
        self.rotrightaround_button.reparentTo(self.interact_frame)

        self.placeitemlog_button = DirectButton(text='log', command=self.placelogfunc, frameSize=(-0.4, 0.4, -0.4, 0.4), text_scale=0.5, sortOrder=1)
        self.placeitemlog_button.setPos(-3.5, 0, -2.5)
        self.placeitemlog_button.reparentTo(self.interact_frame)
        self.placeitempanel_button = DirectButton(text='panel', command=self.placepanelfunc, frameSize=(-0.4, 0.4, -0.4, 0.4), text_scale=0.3, sortOrder=1)
        self.placeitempanel_button.setPos(-3.5, 0, -3.5)
        self.placeitempanel_button.reparentTo(self.interact_frame)
        self.placeitempanel_button = DirectButton(text='stick', command=self.placestickfunc, frameSize=(-0.4, 0.4, -0.4, 0.4), text_scale=0.3, sortOrder=1)
        self.placeitempanel_button.setPos(-3.5, 0, -1.5)
        self.placeitempanel_button.reparentTo(self.interact_frame)
        # self.int_button.setPos(3, 0, -3.15)
        self.interact_frame.hide()
        # self.pause_frame.hide()
        self.pause_frame.reparentTo(self.gui_root) # Parent the frame to aspect2d

        self.resume_button = DirectButton(text=("Resume"), command=self.resume_game, frameSize=(-2.5, 2.5, -0.5, 0.8), text_scale=0.7, sortOrder=1)
        self.resume_button.setPos(0, 0, 0)
        self.resume_button.reparentTo(self.pause_frame)

        self.graphics_button = DirectButton(text=("Settings"), command=lambda: self.key_graphics(), frameSize=(-2.5, 2.5, -0.5, 0.8), text_scale=0.7, sortOrder=1)
        self.graphics_button.setPos(0, 0, -2)
        self.graphics_button.reparentTo(self.pause_frame)

        self.quit_button = DirectButton(text=("QUIT"), command=lambda: self.quit(), frameSize=(-2.5, 2.5, -0.5, 0.8), text_scale=0.7, sortOrder=1)
        self.quit_button.setPos(0, 0, -6)
        self.quit_button.reparentTo(self.pause_frame)

        self.pgraphics_frame = DirectFrame(scale=(120.0 / current_width, 1, 80.0 / current_height), frameColor=(0, 0, 0, 255), frameSize=(-8, 8, -8, 2) )
        self.pgraphics_frame.hide()
        self.pgraphics_frame.reparentTo(self.gui_root)

        self.trent_taray = OnscreenText(text='@trenttaray.bsky.social', frame=None, pos=(0.5, -0.5), scale=0.7, fg=(1, 1, 1, 1))
        self.trent_taray.setPos(0, -4)
        self.trent_taray.reparentTo(self.pause_frame)

        self.res_button = DirectButton(text=("Res=1080x720"), command=lambda: self.change_window_size(1080, 720), frameSize=(-2.5, 2.5, -0.5, 0.8), text_scale=0.7, sortOrder=1)
        self.res_button.setPos(-5, 0, 0)
        self.res_button.reparentTo(self.pgraphics_frame)

        self.resb_button = DirectButton(text=("Res=840x720"), command=lambda: self.change_window_size(840, 720), frameSize=(-2.5, 2.5, -0.5, 0.8), text_scale=0.7, sortOrder=1)
        self.resb_button.setPos(-5, 0, -2)
        self.resb_button.reparentTo(self.pgraphics_frame)

        self.rshowpointer_button = DirectButton(text=('pointer: on'), command=lambda: self.keypointhide(), frameSize=(-1.0, 1.0, -0.5, 0.8), text_scale=0.34, sortOrder=1)
        self.rshowpointer_button.setPos(-6.5, 0, -4)
        self.rshowpointer_button.reparentTo(self.pgraphics_frame)

        self.ralighcursor_button = DirectButton(text=('align: on'), command=lambda: self.keynormals(), frameSize=(-1.0, 1.0, -0.5, 0.8), text_scale=0.34, sortOrder=1)
        self.ralighcursor_button.setPos(-3.5, 0, -4)
        self.ralighcursor_button.reparentTo(self.pgraphics_frame)

        self.resc_button = DirectButton(text=("fullscreen"), command=lambda: self.change_to_fullscreen(), frameSize=(-2.5, 2.5, -0.5, 0.8), text_scale=0.7, sortOrder=1)
        self.resc_button.setPos(-5, 0, -6)
        self.resc_button.reparentTo(self.pgraphics_frame)

        self.resb_button = DirectButton(text=("apply"), command=lambda: self.apply(), frameSize=(-2.5, 2.5, -0.5, 0.8), text_scale=0.7, sortOrder=1)
        self.resb_button.setPos(5, 0, -6)
        self.resb_button.reparentTo(self.pgraphics_frame)

        self.slider = DirectSlider(range=(50, 120), value=100, pageSize=3, command=self.setfovfunc, frameSize=(-2.5, 2.5, -0.5, 0.8))
        self.slider.setPos(5, 0, 0)
        self.slider.reparentTo(self.pgraphics_frame)

        self.fovtext = OnscreenText(text='FOV: value', frame=None, pos=(0.5, -0.5), scale=0.7, fg=(1, 1, 1, 1))
        self.fovtext.setPos(5, 0)
        self.fovtext.reparentTo(self.pgraphics_frame)

        self.sliderVolume = DirectSlider(range=(0, 100), value=3, pageSize=3, command=self.setvolfunc, frameSize=(-2.5, 2.5, -0.5, 0.8))
        self.sliderVolume.setPos(5, 0, -4)
        self.sliderVolume.reparentTo(self.pgraphics_frame)

        self.volumetext = OnscreenText(text='VOLUME: value', frame=None, pos=(0.5, -0.5), scale=0.7, fg=(1, 1, 1, 1))
        self.volumetext.setPos(5, -4)
        self.volumetext.reparentTo(self.pgraphics_frame)

        #instancing
        self.removed_instances = []
        self.total_instances = 0
        self.instance_parent = None

        # Create a new node path to parent the instance to
        self.new_parent_node = render.attachNewNode('new_parent')
        self.new_parent_node.reparentTo(self.render)

        #camera
        lens = camera.node().getLens()
        lens.setNear(0.001)
        lens.setFar(700)

        #misc variables
        self.fs=True
            # Define the base forbidden region
        self.base_forbidden_region = (-300, -300, -1000, 300, 300, 1000)  # (min_x, min_y, min_z, max_x, max_y, max_z)

        self.phpr=[0,0,0]

        self.checkdic={}
        self.objmap = {}
        self.biometn={}
        self.holder = [0,0]
        self.holderb=[0,0]
        self.height_map_xz=[0,0]

        self.height_maps={}
        self.region_maps={}
        self.texture_maps={}
        self.edits={}
        
        self.plantpos={}
        self.dictn={}
        self.numedits=[]
        self.model_data={}
        self.loadedfiles,self.modelnames,self.displaymodels,self.modeldirectory=self.load_bam_files(self.loader, 'data/level/models/', self.shadershadows,self.shader01)
        self.model_data=self.load_bam_filesb(self.loader, 'data/level/models/', self.shadershadows,self.shader01)

        self.all_lists = []

        for file in self.modelnames:
            position_list = []
            scale_list = []
            rotation_list = []

            poswriter_list = None
            scalewriter_list = None
            rotwriter_list = None

            self.all_lists.extend([poswriter_list, scalewriter_list, rotwriter_list, position_list, scale_list, rotation_list])


        # self.primarypositonlisttree=[]
        # self.primaryscalelisttree=[]
        # self.primaryrotationlisttree=[]

        # self.primarypositonlistbush=[]
        # self.primaryscalelistbush=[]
        # self.primaryrotationlistbush=[]
        
        #lighting
        self.sun = DirectionalLight("Spot")

        self.sun_path = self.render.attachNewNode(self.sun)
        self.sun_path.node().set_shadow_caster(True, 4096, 4096)
        self.sun_path.node().set_color((0.9, 0.9, 0.8, 1.0))
        # self.sun_path.node().showFrustum()
        self.sun_path.node().get_lens().set_fov(40)
        # self.sun_path.node().attenuation = (1, 0.001, 0.0001)
        self.sun_path.node().get_lens().set_near_far(-400, 400)
        self.sun_path.node().get_lens().set_film_size(400)

        self.pivot = self.render.attachNewNode("pivot")
        self.pivot.setPos(0, 0, 0)  # Set the position of the pivot point

        self.sun_path.setHpr(0, 0, 0)
        self.sun_path.reparentTo(self.pivot)
        self.render.setLight(self.sun_path)
        #sun shader settings
        self.render.set_shader_input('my_directional_light',self.sun_path)
        self.render.set_shader_input("my_directional_light.direction", self.sun.getDirection())

        
        #ambient 
        self.ambient = AmbientLight('ambient')
        ambient_path = self.render.attachNewNode(self.ambient)
        self.render.setLight(ambient_path)
        #ambient shader settings
        self.render.setShaderInput("ambientLightColor", (0.1, 0.1, 0.1, 1.0))

        #point light
        self.point = PointLight("Point")
        self.point.setAttenuation((0.5, 0.2, 0.02))
        
        self.point.setMaxDistance(20)  # Set the maximum distance of the light's influence
        self.point_path = self.render.attachNewNode(self.point)
        
        # self.point.showFrustum()


        self.spotlight1 = Spotlight("spotlight1")
        self.spotlight1.setColor((10, 10, 10, 1))  # Brighter light (RGB values higher than 1)
        self.spotlight1.setAttenuation((1.0, 0.09, 0.032))
        self.spotlight1.setMaxDistance(90)
        lens = PerspectiveLens()
        lens.setFov(90)  # Adjust this value to change the radius (field of view)
        self.spotlight1.setLens(lens)
        self.flashlight = self.render.attachNewNode(self.spotlight1)
        self.render.setLight(self.flashlight)
        # self.spotlight1.showFrustum()

        self.spotlight2 = Spotlight("spotlight2")
        self.spotlight2.setAttenuation((1.0, 0.09, 0.032))
        self.spotlight2.setMaxDistance(10)
        self.spotlight2_path = self.render.attachNewNode(self.spotlight2)
        self.spotlight2_path.setPos(-22, -22, 26) # Example position
        self.render.setLight(self.spotlight2_path)
        # self.spotlight2.showFrustum()

        self.render.set_shader_input('num_spot_lights', 2)
        self.render.set_shader_input('spot_lights[0]', self.flashlight)
        self.render.set_shader_input('spot_lights[1]', self.spotlight2_path)
        # self.flashlight.reparentTo(self.point_path)
        
        # self.point_path.reparentTo(self.flashlight)
        
        self.render.set_shader_input(f'spot_lights[1].color', LVecBase4(0, 0, 0, 0))#off
        self.render.set_shader_input(f'spot_lights[0].color', LVecBase4(0, 0, 0, 0))#off
        for i in range(2,4):
            self.render.set_shader_input(f'spot_lights[{i}]',self.flashlight)
            self.render.set_shader_input(f'spot_lights[{i}].color', LVecBase4(0, 0, 0, 0))#off
        # Initialize point lights
        self.point1 = PointLight("Point1")
        self.point1.setAttenuation((1.0, 0.09, 0.032))
        self.point1.setMaxDistance(50)
        self.point1_path = self.render.attachNewNode(self.point1)
        self.point1_path.setPos(-16, -31, 29) # Example position
        self.render.setLight(self.point1_path)

        self.point2 = PointLight("Point2")
        self.point2.setAttenuation((0.1, 0.43, 0.044))
        self.point2.setMaxDistance(2)
        self.point2_path = self.render.attachNewNode(self.point2)
        self.point2_path.setPos(-22, -22, 29) # Example position
        self.render.setLight(self.point2_path)

        self.render.set_shader_input('num_point_lights', 2)
        # Set shader inputs for existing lights
        self.render.set_shader_input('point_lights[0]',self.point_path)#flashlightpoint
        
        self.render.set_shader_input('point_lights[1]',self.point2_path)
        self.render.set_shader_input('point_lights[1].color', LVecBase4(0, 0, 0, 0))#off
        self.render.set_shader_input('point_lights[0].color', LVecBase4(0, 0, 0, 0))#off
        for i in range(2,4):
            self.render.set_shader_input(f'point_lights[{i}]',self.point1_path)
            self.render.set_shader_input(f'point_lights[{i}].color', LVecBase4(0, 0, 0, 0))#off

        # Update num_point_lights

        self.xh,self.zh=0,0
        

        # # Mount the multifile to the VFS
        vfs = VirtualFileSystem.getGlobalPtr()
        mf = Multifile()
        mf.openRead("data/level/packfiles.mf")
        vfs.mount(mf, '.', VirtualFileSystem.MFReadOnly)
        getModelPath().appendDirectory('.')
        getModelPath().appendDirectory("data/level/")


        #file management WIP
        # parent_folder = 'data/level/terrains'
        # parent_folder = os.path.join(os.path.dirname(__file__), 'data/level/terrains')

        parent_folder = 'data/level/terrains'
        random_folder = self.choose_random_folder(parent_folder)

        folder_path = 'data/level/terrains/' + random_folder
        random_files = self.choose_random_files(folder_path, 1)
        random_files_no_ext = [os.path.splitext(f)[0] for f in random_files]
    
        # Initialize an array with the image data directly
        self.brush_height2im= PNMImage()
        self.brush_height2im.read(Filename('data/level/terrains/' + random_folder+'/'+ random_files[0]))
        
        # Convert PNMImage to NumPy array and make it grayscale
        width = self.brush_height2im.get_x_size()
        height = self.brush_height2im.get_y_size()

        self.brush_height2 = np.array([[self.brush_height2im.get_gray(x, y) * 255 for x in range(width)] for y in range(height)], dtype=np.uint8)
        # self.brush_height2 = [ [int(self.brush_height2im.getGray(x, y) * 255 * 0.1) for x in range(width)] for y in range(height) ]#[ [int(self.brush_height2im.getGray(x, y) * 255) for x in range(width)] for y in range(height) ]
        self.bigtest = PNMImage()
        self.bigtest.read(Filename('data/level/terrains/' + random_folder+'/'+ random_files_no_ext[0]+'.png'))

        self.regionfile01 = PNMImage()
        self.regionfile01.read(Filename('data/level/terrains/mountains/brush1region.png'))
        
        # Convert PNMImage to NumPy array and make it grayscale
        widthr = self.regionfile01.get_x_size()
        heightr = self.regionfile01.get_y_size()

        self.regionfile01 = np.array([[self.regionfile01.get_gray(x, y) * 255 for x in range(widthr)] for y in range(heightr)], dtype=np.uint8)

        # Load the smaller image
        # self.brushshovel = np.load('data/level/handbrushes/mountains/brushshovel1.npy')

        # Read the brush height map image
        brush_height2ims = PNMImage()
        brush_height2ims.read(Filename('data/level/handbrushes/mountains/brushshovel1map.png'))
        
        # Convert PNMImage to NumPy array and make it grayscale
        widths = brush_height2ims.get_x_size()
        heights = brush_height2ims.get_y_size()

        # Initialize an array with the image data directly
        # self.brushshovel = [ [int(brush_height2im.getGray(x, y) * 255) for x in range(width)] for y in range(height) ]
        # [ [int(brush_height2im.getGray(x, y) * 255 * 0.1) for x in range(width)] for y in range(height) ]
        # self.brushshovel = [ [int(brush_height2ims.getGray(x, y) * 255 * 0.1) for x in range(widths)] for y in range(heights) ]
        self.brushshovel = np.array([[brush_height2ims.get_gray(x, y) * 255 for x in range(widths)] for y in range(heights)], dtype=np.uint8)
        self.small_imagetest = PNMImage()
        self.small_imagetest.read(Filename('data/level/handbrushes/mountains/brushshovel1.png'))
   
        self.textureC = Texture()
        
        #loading process
        self.filelist=[]


        #instance objects
        # self.tree = self.loader.loadModel("tree1.bam")
        # self.tree.hide()
        # self.tree.setScale(1)
        # self.tree.setShader(self.shader01)
        # self.tree.reparentTo(render)
        # self.tree.setTwoSided(True)
        # self.tree.setTransparency(TransparencyAttrib.MAlpha)


        # self.bush = self.loader.loadModel("burdock0.bam")
        # self.bush.setScale(1)
        # self.bush.setShader(self.shadershadows)
        # self.bush.reparentTo(render)
        # self.bush.setTwoSided(True)
        # self.bush.setTransparency(TransparencyAttrib.MAlpha)


        #testing objects
        # self.monkey1 = self.loader.loadModel('my-models/monkey')
        # self.monkey2 = self.loader.loadModel('my-models/monkey')

        # self.monkey1.reparentTo(render)
        # self.monkey1.setPos(0, 0, 16)

        # self.monkey2.reparentTo(render)
        # self.monkey2.setPos(1, -1, 17)

        # self.monkey1.setShader(self.shader01)
        # self.monkey2.setShader(self.shader01)

        # Load the skybox models
        self.day_skybox = self.loader.loadModel("dayskybox.bam")
        # self.night_skybox = self.loader.loadModel("nightskybox.bam")
        self.day_skybox.setTransparency(TransparencyAttrib.MAlpha)
        # self.night_skybox.setTransparency(TransparencyAttrib.MAlpha)
        self.moverskybox = self.render.attachNewNode("moverskybox")
        self.day_skybox.setScale(2000)  # Adjust scale as needed
        self.day_skybox.reparentTo(render)
        self.day_skybox.setShaderOff()
        self.day_skybox.setLightOff()
        self.day_skybox.setBin('background', 0)
        self.day_skybox.setDepthWrite(0)
        self.day_skybox.setShader(self.shader01)
        self.day_skybox.reparentTo(self.moverskybox)


        # self.night_skybox.setScale(2000)  # Adjust scale as needed
        # self.night_skybox.reparentTo(render)
        # self.night_skybox.setShaderOff()
        # self.night_skybox.setLightOff()
        # self.night_skybox.setBin('background', 0)
        # self.night_skybox.setDepthWrite(0)
        # self.night_skybox.setShader(self.shader01)
        # self.night_skybox.reparentTo(self.moverskybox)

        # self.night_skybox.setScale(2000)  # Adjust scale as needed
        # self.night_skybox.reparentTo(render)
        # self.night_skybox.setShaderOff()
        # self.night_skybox.setLightOff()
        # self.night_skybox.setBin('background', 0)
        # self.night_skybox.setDepthWrite(0)
        # self.night_skybox.setShader(self.shader01)
        # self.night_skybox.reparentTo(self.moverskybox)
        noise_height_map = np.random.rand(4486, 4486)

        self.cloud = self.loader.loadModel("cloud0.bam")
        self.cloud.setShader(self.shadershadows)
        self.cloud.setTwoSided(True)
        self.cloud.setTransparency(TransparencyAttrib.MAlpha)


        # self.cloud.setPos(92.76170349121094,-17.836923599243164,0)
        self.cloud.reparentTo(self.render)
        tex = self.cloud.findTexture("*")

        texture_path = os.path.join('data/level/cloud_atlas.png')
        if os.path.exists(texture_path):
            tex.read(texture_path)
        else:
            print(f"Texture file not found: {texture_path}")
        

       
        self.pivot.reparentTo(self.moverskybox)

        # self.apply_force_field(self.height_maps[tuple([0,0])],instances.getChildren(),Vec3(0, 0, 0),19000,200,'attract')

        # enable physics
        base.enableParticles()
        base.cTrav = CollisionTraverser("base collision traverser")
        base.cTrav.setRespectPrevTransform(True)
        # base.cTrav.showCollisions(render)

        # setup default gravity
        gravityFN = ForceNode("world-forces")
        # gravityFNP = render.attachNewNode(gravityFN)
        gravityForce = LinearVectorForce(0, 0, -9.81)
        gravityFN.addForce(gravityForce)
        base.physicsMgr.addLinearForce(gravityForce)
        self.pointer = self.loader.loadModel("data/level/pointer.bam")
        self.pointer.reparentTo(self.render)
        self.pointer.setScale(0.2)
        self.pointer.setShader(self.shader01)
        
        # # # Intangible blocks (as used for example for collectible or event spheres)
        self.moveThroughBoxesb = render.attachNewNode(CollisionNode("Ghosts"))
        self.boxb = CollisionBox((0, 0, 1), 1, 1, 1)
        self.boxb.setTangible(False)
        self.moveThroughBoxesb.node().addSolid(self.boxb)
        self.moveThroughBoxesb.node().setFromCollideMask(BitMask32.allOff())
        self.moveThroughBoxesb.node().setIntoCollideMask(BitMask32(0x10))

        # self.moveThroughBoxesb.show()

        self.moveThroughBoxesh = render.attachNewNode(CollisionNode("Ghosts"))
        self.boxh = CollisionBox((0, 0.0, 0.5), 1, 1, 0.5)
        self.boxh.setTangible(False)
        self.moveThroughBoxesh.node().addSolid(self.boxh)
        self.moveThroughBoxesh.node().setFromCollideMask(BitMask32.allOff())
        self.moveThroughBoxesh.node().setIntoCollideMask(BitMask32(0x10))

        # self.moveThroughBoxesh.show()

        self.moveThroughBoxesh1 = render.attachNewNode(CollisionNode("Ghosts"))
        self.boxh1 = CollisionBox((0, -0.9, 0.5), 1, 0.1, 0.5)
        self.boxh1.setTangible(False)
        self.moveThroughBoxesh1.node().addSolid(self.boxh1)
        self.moveThroughBoxesh1.node().setFromCollideMask(BitMask32.allOff())
        self.moveThroughBoxesh1.node().setIntoCollideMask(BitMask32(0x10))

        # self.moveThroughBoxesh1.show()

        self.moveThroughBoxesh2 = render.attachNewNode(CollisionNode("Ghosts"))
        self.boxh2 = CollisionBox((0, 0.9, 0.5), 1, 0.1, 0.5)
        self.boxh2.setTangible(False)
        self.moveThroughBoxesh2.node().addSolid(self.boxh2)
        self.moveThroughBoxesh2.node().setFromCollideMask(BitMask32.allOff())
        self.moveThroughBoxesh2.node().setIntoCollideMask(BitMask32(0x10))

        # self.moveThroughBoxesh2.show()

        self.moveThroughBoxesh3 = render.attachNewNode(CollisionNode("Ghosts"))
        self.boxh3 = CollisionBox((-0.9, 0, 0.5), 0.1, 1, 0.5)
        self.boxh3.setTangible(False)
        self.moveThroughBoxesh3.node().addSolid(self.boxh3)
        self.moveThroughBoxesh3.node().setFromCollideMask(BitMask32.allOff())
        self.moveThroughBoxesh3.node().setIntoCollideMask(BitMask32(0x10))

        # self.moveThroughBoxesh3.show()

        self.moveThroughBoxesh4 = render.attachNewNode(CollisionNode("Ghosts"))
        self.boxh4 = CollisionBox((0.9, 0, 0.5), 0.1, 1, 0.5)
        self.boxh4.setTangible(False)
        self.moveThroughBoxesh4.node().addSolid(self.boxh4)
        self.moveThroughBoxesh4.node().setFromCollideMask(BitMask32.allOff())
        self.moveThroughBoxesh4.node().setIntoCollideMask(BitMask32(0x10))

        # self.moveThroughBoxesh4.show()

        # self.moveThroughBoxes.setPos(-26,48,-88)

        # self.accept("CharacterCollisions-in-Ghosts", print, ["ENTER"])
        # self.accept("CharacterCollisions-out-Ghosts", print, ["EXIT"])


        # Create a CollisionTraverser
        self.traverser = CollisionTraverser()

        # Create a CollisionHandlerQueue to hold the collision entries
        self.queuec = CollisionHandlerQueue()
        self.rayC = CollisionRay()
        self.rayC.setOrigin(Point3(0, 0, 0))  # Set the origin of the ray
        self.rayC.setDirection(Vec3(0, 0, -1))  # Set the direction of the ray
        cnodeC = CollisionNode('rayNode')
        cnodeC.addSolid(self.rayC)
        # Combine bit 1 and 0x80 using bitwise OR
        collision_mask = BitMask32.bit(1)
        # Set the combined collision mask
        cnodeC.setFromCollideMask(collision_mask)
        cnodeC.setIntoCollideMask(BitMask32.allOff())
        
        # Attach the CollisionNode to a NodePath and add it to the traverser
        rayNodePathC = render.attachNewNode(cnodeC)
        self.traverser.addCollider(rayNodePathC, self.queuec)


        self.queueb = CollisionHandlerQueue()
        self.ray01 = CollisionRay()
        self.ray01.setOrigin(0, 0, 0)
        self.ray01.setDirection(0, 0, -1)

        rayNode01 = CollisionNode('ray')
        rayNode01.addSolid(self.ray01)
        rayNodePath = render.attachNewNode(rayNode01)

        rayNode01.setFromCollideMask(BitMask32(0x10))
        rayNode01.setIntoCollideMask(BitMask32.allOff())

        self.traverser.addCollider(rayNodePath , self.queueb)

        self.queue = CollisionHandlerQueue()
        self.ray = CollisionRay()
        self.ray.setOrigin(Point3(0, 0, 0))  # Set the origin of the ray
        self.ray.setDirection(Vec3(0, 0, -1))  # Set the direction of the ray
        cnode = CollisionNode('rayNode')
        cnode.addSolid(self.ray)
        # Combine bit 1 and 0x80 using bitwise OR
        collision_mask = BitMask32.bit(1)
        # Set the combined collision mask
        cnode.setFromCollideMask(collision_mask)
        cnode.setIntoCollideMask(BitMask32.allOff())

        # Attach the CollisionNode to a NodePath and add it to the traverser
        rayNodePath = render.attachNewNode(cnode)
        self.traverser.addCollider(rayNodePath, self.queue)
        # Set the world
        self.world = base.cTrav
        self.filename = "playerdata.json"
        self.worldname='world1'
        # Check if the file exists
        if not os.path.exists(self.filename):
            # If the file does not exist, create it with default data
            default_data = {"name": "Player", "fov": 100, "volume": 20, "pointer": 0,"align": 0}
            self.data_loaded=default_data
            with open(self.filename, "w") as file:
                json.dump(default_data, file, indent=4)
        else:
            # If the file exists, load the data
            with open(self.filename, "r") as file:
                self.data_loaded = json.load(file)

        
        self.filename_world = "saves/"+self.worldname+"/playerdata.json"
        # Check if the file exists
        if not os.path.exists(self.filename_world):
            # If the file does not exist, create it with default data
            default_data = {"name": "Player", "position": [0,0,0], "facing":[0,0,0], "time":[0,0]}
            self.world_data_player=default_data
            with open(self.filename_world, "w") as file:
                json.dump(default_data, file, indent=4)
        else:
            # If the file exists, load the data
            with open(self.filename_world, "r") as file:
                self.world_data_player = json.load(file)


        int_value = round(self.data_loaded["fov"])

        self.fovtext.setText(f'FOV {int_value}')
        base.camLens.setFov(int_value)
        self.slider["value"] = int_value
        int_value = round(self.data_loaded["volume"])
        self.keypointhidetog=True
        if self.data_loaded["pointer"] == 1:

            if self.keypointhidetog == False:
                self.keypointhide()
        else:

            if self.keypointhidetog == True:
                self.keypointhide()
        self.keynormalstog=True
        if self.data_loaded["align"] == 1:

            if self.keynormalstog == False:
                self.keynormals()
        else:

            if self.keynormalstog == True:
                self.keynormals()

        # self.fovtext.setText(f'{int_value}')
        # base.camLens.setFov(int_value)
        self.sliderVolume["value"] = int_value
        #player
        self.player = PlayerController(self.world, "data/config.json")
        self.player.startPlayer()
        plhpr=self.world_data_player["facing"]
        self.player.setStartHpr(Vec3(plhpr[0],plhpr[1],plhpr[2]))
        self.player.setShader(self.shader01)
        

        # self.loading_done=True
        player_pos = self.player.main_node.getPos()

        self.flashlight.setPos(player_pos[0],player_pos[1],player_pos[2])
        # self.flashlight.setPos(0,0,-1)
        self.flashlight.setHpr(0,0,0)

        self.flashlight.reparentTo(base.camera)



        self.total_instances1 = 0
        self.instance_parent1 = None
        self.poswriter1=None
        self.rotwriter1=None
        self.scalewriter1=None
        positionlist=self.position_gen(1000,12,4486,4486,noise_height_map)
        (self.total_instances1, 
        self.poswriter1, 
        self.rotwriter1, 
        self.scalewriter1,
        self.offsetpositionlist, 
        self.scalelist, 
        self.rotationlist) = self.setup_instancingcloud(
            noise_height_map,
            self.cloud, 
            positionlist, 
            fromvalue=0, 
            tovalue=50, 
            new_location=(player_pos[0], player_pos[1], 90), 
            zlevel=None, 
            total_instances=self.total_instances1, 
            poswriter=self.poswriter1,
            rotwriter=self.rotwriter1,
            scalewriter=self.scalewriter1,
            zrotonly=True
        )
        self.sun_path.setPos(player_pos[0],player_pos[1]-180, player_pos[2])
        self.xz=[0,0]
        self.xz0=[0,0]
        self.xz1=[0,0]
        
        self.player.flymodeon()
        self.player.main_node.setPos(self.world_data_player["position"][0], self.world_data_player["position"][1], self.world_data_player["position"][2])

        # self.ladder = self.loader.loadModel('data/level/ladder.bam')
        # self.ladder.reparentTo(render)
        # self.ladder.setPos(0, 0, 40)

        # self.ladder.setTransparency(TransparencyAttrib.MAlpha)
        xh = math.ceil(self.player.main_node.getPos()[0] / 54) * 54
        zh = math.ceil(self.player.main_node.getPos()[1] / 54) * 54

        self.xpos,self.zpos,_=self.player.main_node.getPos()
        x,z=int(self.xpos),int(self.zpos)

        self.xz0[0]=x
        self.xz0[1]=z
        self.xz[0]=x
        self.xz[1]=z

        self.holder = [xh, zh]

        # Define the range for checking around the coordinate
        self.prev_x = xh
        self.prev_z = zh
        self.x_positive = x + 54
        self.z_positive = z + 54
        self.x_negative = x - 90
        self.z_negative = z - 90
        # Define the range for checking around the coordinate
        range_x = 2  # Number of tiles to check in the x direction
        range_z = 2  # Number of tiles to check in the z direction
        tile_size = 54  # Size of each tile

        # Generate terrain in a grid pattern around the current position
        # Add the update task
        # Set up the event listener for the left mouse button
        base.accept('mouse1', self.performRaycastLeft)
        base.accept('mouse3', self.performRaycastRight)

        self.rotates=0
        # Define colors for different times of the day
        self.sunrise_color = Vec4(1, 0.5, 0.5, 1)  # Soft red
        self.day_color = Vec4(0.8, 0.8, 0.7, 1)        # Bright yellowish-white
        self.sunset_color = Vec4(0.5, 0.2, 0.2, 1) # Dark red
        self.night_color = Vec4(0.02, 0.02, 0.02, 1)  # Dark blue
        self.night_counter = 0  # Counter to keep track of the nights

        # Define your horizon color
        self.horizon_colorday = Vec4(0.529, 0.808, 0.980, 1)
        self.horizon_colornight = Vec4(0.02, 0.02, 0.02, 1)
        # Set the shader input for the horizon color
        self.render.setShaderInput("horizonColorb", self.horizon_colorday)

        self.fogday_color = Vec4(0.5, 0.5, 0.5, 1.0)
        self.fognight_color = Vec4(0.02, 0.02, 0.02, 1.0)

        self.timescale=2.0
        self.update_night = True  # Flag to control the update
        self.moon_phase_counter = 0  # Counter to keep track of the moon phases
        self.custom_time = 120.0  # Initialize your custom time variable
        self.moon_phase_counter = self.world_data_player["time"][1]
        self.custom_time = self.world_data_player["time"][0]
        self.genter_scheduled = False

        self.incre=0
        self.rotationsmem=[0,0,0]
        self.zmem=0
        
        self.world_name_n = "saves/"+self.worldname+"/world.json"
        self.world_name_p = "saves/"+self.worldname+"/placedata.json"
        self.world_name_r = "saves/"+self.worldname+"/removeddata.json"

        self.itemdex="panel0_0_0_0_1_2"
        self.distance_moved=0
        self.nodedict={}
        self.forremoval={}
        self.heldentity=''
        self.clicked=False

        self.bufferdict2=[]
        self.forremoval = self.save_dict_as_json(self.world_name_r,self.forremoval, self.bufferdict2)
        
        self.bufferdict3=[]
        self.forplacement={}
        self.forplacement = self.save_dict_as_json(self.world_name_p,self.forplacement, self.bufferdict3)
        self.forplacementready=copy.deepcopy(self.forplacement)

        self.bufferdict=[]
        self.savesedit={}
        self.savesedit=self.save_dict_as_json(self.world_name_n,self.savesedit, self.bufferdict)

        self.bufferdictbrush=[]
        self.todisplay=None

        self.itemdex = "panel0_0_0_0_1_2"
        if self.todisplay:
            self.todisplay.hide()
        self.todisplay = None
        self.update_display_item()

        self.ld = 0
        self.range_x = 2
        self.range_z = 2
        self.tile_size = 270#54
        # Load the background image
        bg_image = self.loader.loadTexture("data/backgroundimage.png")
        
        # Create a DirectFrame with the background image
        self.bg_frame = DirectFrame(frameSize=(-1.5, 1.5, -1, 1),
                                    frameTexture=bg_image,
                                    frameColor=(1, 1, 1, 1))
        self.bg_frame.setTransparency(TransparencyAttrib.MAlpha)
        loadbar_image = self.loader.loadTexture("data/loadingbar.png")
        self.loadingBar = DirectWaitBar(text="", value=0, pos=(0, 0.5, -0.5), frameTexture=loadbar_image,barColor=(1, 1, 0, 0.5) )
        self.prev_frame = -1
        self.count=0
        # self.holding.setPos(0, -4)
        # Start the terrain initialization task
        # self.player.flymodeon()
        # self.npc = self.loader.loadModel("cube.bam")
        # self.npc.reparentTo(self.render)
        # collision_node = CollisionNode('npcCollision')
        # collision_node.addSolid(CollisionSphere(0, 0, 0, 1))
        # self.npc.attachNewNode(collision_node)
        # self.npc.setPos(-1007, -507, 3)
        # self.npc_direction = Vec3(1, 0, 0)  # Moving along the X-axis
        # self.npc_speed = 5  # Units per second

        self.taskMgr.doMethodLater(0.5,self.initialize_terrain_task, "initialize_terrain_task")

        # for x_offset in range(-range_x, range_x + 1):
        #     for z_offset in range(-range_z, range_z + 1):
        #         self.x_check = self.prev_x + x_offset * tile_size
        #         self.z_check = self.prev_z + z_offset * tile_size
        #         self.generate_terrain_if_needed(self.x_check, self.z_check)
        # player_pos = self.player.main_node.getPos()
        # hmap0=(int(player_pos[0]  // 486) * 486,int(player_pos[1] // 486) * 486)
        # heightmapsgot = self.height_maps.get((hmap0))
        # ground_level=heightmapsgot[int(player_pos[0]%486)][int(player_pos[1]%486)]
        # self.player.main_node.setPos(player_pos[0],player_pos[1],ground_level+0.1)
        # self.taskMgr.doMethodLater(0.3,self.genter, 'genter', taskChain='terrain_chain')#, extraArgs=[prev_x, x_positive, prev_z, z_positive, x_negative, z_negative], appendTask=True
        # self.taskMgr.doMethodLater(0.1, self.process_files, "ProcessFilesTask")
        # self.task = self.taskMgr.add(self.updateTask, "update")


    # Function to set master volume
    def set_master_volume(self, volume):
        self.master_volume = volume/4
        self.audio_mgr.setVolume(self.master_volume)


    def initialize_terrain_task(self, task):
        if self.pause == False:
            self.togglePause()
        if self.ld < (self.range_x * 2 + 1) * (self.range_z * 2 + 1):
            x_offset = (self.ld // (self.range_z * 2 + 1)) - self.range_x
            z_offset = (self.ld % (self.range_z * 2 + 1)) - self.range_z
            self.x_check = self.prev_x + x_offset * self.tile_size
            self.z_check = self.prev_z + z_offset * self.tile_size
            self.generate_terrain_if_needed(self.x_check, self.z_check)
            self.ld += 1
            self.loadingBar["value"] = (self.ld / ((self.range_x * 2 + 1) * (self.range_z * 2 + 1))) * 100
            
            # Schedule genter only if it hasn't been scheduled already
            # if not self.genter_scheduled:
            #     self.genter_scheduled = True
            

            return Task.cont
        else:
            if not self.genter_scheduled:
                self.taskMgr.doMethodLater(0.3, self.genter, 'genter', taskChain='terrain_chain')
                self.genter_scheduled = True

            if len(self.storework) > 0:
                return Task.cont  # Continue the task until self.storework is empty

            # Perform the rest of your code once self.storework is empty
            self.loadingBar.hide()
            self.bg_frame.hide()
            # if self.pause == True:
            #     self.togglePause()
            self.countbb = 0
            self.player.flymodeoff()
            self.taskMgr.doMethodLater(0.1, self.process_files, "ProcessFilesTask")
            self.task = self.taskMgr.add(self.updateTask, "update")
            # self.taskMgr.doMethodLater(0.1, self.updateNPC, "UpdateNPC")

            return Task.done




    def updateTask(self, task):
        # print(self.player.getCurrentAnim())
        # print(self.player.getCurrentFrame())
        self.count+=0.1
        visible_instances1 = self.moveclouds(self.poswriter1, self.scalewriter1, self.rotwriter1, self.offsetpositionlist, self.rotationlist, self.scalelist, camera=base.camera, radius=1400)
        if self.key2tog and not self.keyCtog:
            self.raycastangle()
        if self.key3tog==True:
            self.raycastlight()
        player_pos = self.player.main_node.getPos()
        hmap0=(int(player_pos[0]  // 486) * 486,int(player_pos[1] // 486) * 486)
        heightmapsgot = self.height_maps.get((hmap0))
        threshold = -100
        ground_level=heightmapsgot[int(player_pos[0]%486)][int(player_pos[1]%486)]
        if player_pos.getZ() < ground_level+threshold:
            self.player.main_node.setZ(ground_level+1)
        # self.pivot.setPos(player_pos[0],player_pos[1], player_pos[2])
        self.moverskybox.setPos(player_pos[0],player_pos[1], player_pos[2])
        # Calculate the angle of the sun
        angle = (self.custom_time % 360 * self.timescale) % 360


        self.custom_time += 0.001

        # self.shadow_cam.setPos(player_pos[0],player_pos[1]-20, player_pos[2]+30)

        # Determine the current color based on the angle
        if 0 <= angle < 180:
            # self.sun_path.setHpr(0, -angle, 0)
            self.pivot.setHpr(0, -angle, 0)
            # self.render.set_shader_input(f'spot_lights[0].color', LVecBase4(0, 0, 0, 0))#off
            # self.shadow_cam.setHpr(0, -angle, 0)
            # Daytime: make the day skybox opaque and the night skybox transparent

        else:
            # self.sun_path.setHpr(0, angle, 0)
            self.pivot.setHpr(0, angle, 0)
            # self.shadow_cam.setHpr(0, angle, 0)
            # self.render.set_shader_input(f'spot_lights[0].color', LVecBase4(1, 1, 1, 1))#on
            # Nighttime: make the night skybox opaque and the day skybox transparent

        if 0 <= angle < 30:
            # Transition from night to sunrise
            ratio = angle / 30
            self.current_color = self.night_color + (self.sunrise_color - self.night_color) * ratio
            self.fogcurrent_color = self.fognight_color + (self.fogday_color - self.fognight_color) * ratio
            self.horizonfogcurrent_color = self.horizon_colornight + (self.horizon_colorday - self.horizon_colornight) * ratio
            # self.render.set_shader_input(f'spot_lights[0].color', LVecBase4(0, 0, 0, 0))#off
            # day_transparency = 0  # Night skybox is fully visible
            # night_transparency = 1 - ratio  # Fade out the night skybox

        elif 30 <= angle < 180:
            # Transition from sunrise to day
            ratio = (angle - 30) / 150
            self.current_color = self.sunrise_color + (self.day_color - self.sunrise_color) * ratio
            self.fogcurrent_color = self.fogday_color
            self.horizonfogcurrent_color = self.horizon_colorday
            # self.render.set_shader_input(f'spot_lights[0].color', LVecBase4(0, 0, 0, 0))#off
            # day_transparency = ratio  # Fade in the day skybox
            # night_transparency = 0  # Night skybox is fully transparent

        elif 180 <= angle < 210:
            # Transition from day to sunset
            ratio = (angle - 180) / 30
            self.current_color = self.day_color + (self.sunset_color - self.day_color) * ratio
            self.fogcurrent_color = self.fogday_color + (self.fognight_color - self.fogday_color) * ratio
            self.horizonfogcurrent_color = self.horizon_colorday + (self.horizon_colornight - self.horizon_colorday) * ratio
            # self.render.set_shader_input(f'spot_lights[0].color', LVecBase4(0, 0, 0, 0))#off
            # day_transparency = 1 - ratio  # Fade out the day skybox
            # night_transparency = 0  # Night skybox is fully transparent

        elif 210 <= angle < 360:
            # Transition from sunset to night
            ratio = (angle - 210) / 150
            self.current_color = self.sunset_color + (self.night_color - self.sunset_color) * ratio
            self.fogcurrent_color = self.fognight_color
            self.horizonfogcurrent_color = self.horizon_colornight
            # self.render.set_shader_input(f'spot_lights[0].color', LVecBase4(1, 1, 1, 1))#on
            # day_transparency = 0  # Day skybox is fully transparent
            # night_transparency = ratio  # Fade in the night skybox

        # Determine the brightness based on the moon phase
        if self.moon_phase_counter < 15:
            # New moon (darker nights)
            moon_brightness = max(0, 1 - (self.moon_phase_counter / 15.0))
        elif self.moon_phase_counter < 30:
            # Full moon (brighter nights)
            moon_brightness = min(1, (self.moon_phase_counter - 15) / 15.0)
        else:
            # Reset the moon phase counter after one full cycle
            self.moon_phase_counter = 0


        if angle >= 180:
            self.current_color = Vec4(self.current_color.x * moon_brightness, self.current_color.y * moon_brightness, self.current_color.z * moon_brightness, 1)


        self.ambient.setColor(self.current_color)
        self.sun.setColor(self.current_color)
        self.render.setShaderInput("ambientLightColor", self.current_color)

        # self.day_skybox.setColor(1, 1, 1, day_transparency)
        # self.night_skybox.setColor(1, 1, 1, night_transparency)
        
        self.render.setShaderInput("fogColor", self.fogcurrent_color)
        self.render.setShaderInput("horizonColorb", self.horizonfogcurrent_color)
        # Update the moon phase counter at the end of each day
        if angle >= 359 and self.update_night:
            self.moon_phase_counter += 1
            self.update_night = False  # Reset the flag

        # Reset the flag when a new day starts
        if angle < 1:
            self.update_night = True

        self.xpos,self.zpos,ypos=self.player.main_node.getPos()
        x,z=int(self.xpos),int(self.zpos)

        self.prev_x = self.holder[0]
        self.prev_z = self.holder[1]

        dxb = x - self.xz0[0]
        dzb = z - self.xz0[1]
        distance_movedb = math.sqrt(dxb**2 + dzb**2)
        if distance_movedb > 50:

            # Example of updating the position
            # new_position = [10, 20, 30]
            self.world_data_player["position"] = [self.xpos,self.zpos,ypos]
            plhprget=self.player.plugin_getHpr()
            self.world_data_player["facing"] = [plhprget[0],plhprget[1],plhprget[2]]
            self.world_data_player["time"][0]=self.custom_time % 360
            self.world_data_player["time"][1]=self.moon_phase_counter
            # Write the updated data back to the JSON file
            with open(self.filename_world, "w") as file:
                json.dump(self.world_data_player, file, indent=4)



            self.xz0[0]=x
            self.xz0[1]=z

            self.taskMgr.add(self.removeradius, 'removeradius', appendTask=True)


        # # Determine the direction of movement
        self.x_positive = x + 54
        self.x_negative = x - 90

        self.z_positive = z + 54
        self.z_negative = z - 90
        dx = x - self.xz[0]
        dz = z - self.xz[1]
        self.distance_moved = math.sqrt(dx**2 + dz**2)
            # self.thread = Thread(target=self.background_task)
            # self.thread.start()
        if self.distance_moved > 5:
            self.xz[0]=x
            self.xz[1]=z
            # self.check_and_genter(prev_x, x_positive,prev_z,z_positive,x_negative,z_negative)
            # self.clearing=False
            # self.activatecont += 1


            # self.taskMgr.add(self.genter, 'genter', taskChain='terrain_chain')
            
            # Generate terrain for each step the player has moved
            # for x_check in range(self.prev_x, self.x_positive, 54):
            #     for z_check in range(self.prev_z, self.z_positive, 54):
            #         self.generate_terrain_if_needed(x_check, z_check)

            # for x_check in range(self.prev_x, self.x_negative, -54):
            #     for z_check in range(self.prev_z, self.z_negative, -54):
            #         self.generate_terrain_if_needed(x_check, z_check)
            
            # self.taskMgr.add(self.make_instances, 'makeinstances', taskChain='make_instances')
        return task.cont

    @pstat_collector("App:process files")
    def process_files(self, task):
        
        for i, item in enumerate(self.loadedfiles):
            index = i * 6  # Calculate the base index for each file.
            poswriter_list = self.all_lists[index]
            rotwriter_list = self.all_lists[index+1]
            scalewriter_list = self.all_lists[index+2]

            visible_instances = self.cull_and_update_instances(self.loadedfiles[i][0], base.cam, 40.0, poswriter_list, scalewriter_list, rotwriter_list, max_distance=450)
            if visible_instances < 1:
                self.loadedfiles[i][0].hide()
            else:
                self.loadedfiles[i][0].show()
                self.loadedfiles[i][0].setInstanceCount(visible_instances)
        
        # Return task.cont to continue running this task
        return task.again

    def updateNPC(self, task):
        dt = globalClock.getDt()  # Get the time elapsed since the last frame

        # Calculate the direction vector from NPC to player
        npc_pos = self.npc.getPos()
        player_pos = self.player.main_node.getPos()
        # print(player_pos)
        direction = player_pos - npc_pos
        direction.normalize()  # Normalize the direction vector
        hmap0=(int(npc_pos[0]  // 486) * 486,int(npc_pos[1] // 486) * 486)
        heightmapsgot = self.height_maps.get((hmap0))

        ground_level=heightmapsgot[int(npc_pos[0]%486)][int(npc_pos[1]%486)]
        # print(ground_level)
        # Update the NPC's position
        new_pos = npc_pos + direction * self.npc_speed * dt
        self.npc.setPos(new_pos)
        # if npc_pos.getZ() > ground_level+2:
        self.npc.setZ(ground_level+1)

        return task.again



    


    def moveclouds(self, poswriter, scalewriter, rotwriter, positionlist, rotationlist, scalelist, camera, radius=500):
        poswriter.setRow(0)
        rotwriter.setRow(0)
        scalewriter.setRow(0)
        visible_instances = 0
        cam_pos = camera.getPos(render)
        # print(cam_pos)
        # print(self.total_instances1)
        for i, (pos, rotation, scale) in enumerate(zip(positionlist, rotationlist, scalelist)):
            # Move cloud and apply wrapping using modulo relative to camera position
            instance_position = LPoint3f((pos[0] + self.count - cam_pos.getX()) % (radius) - radius,
                                        (pos[1] - cam_pos.getY()) % (radius) - radius,
                                        90)
            
            # Adjust position to wrap around the camera's position
            instance_position += (cam_pos[0]+487,cam_pos[1]+487,90)
            
            poswriter.add_data3(instance_position)
            rotwriter.add_data4(LVecBase4f(rotation[0], rotation[1], rotation[2], 1))
            scalewriter.add_data4(LVecBase4f(scale[0], scale[1], scale[2], 1))
            visible_instances += 1
            
        return visible_instances

        
        # Return task.cont to continue running this task


    # def cullTask(self):
    #     while True:


    #         for i, item in enumerate(self.loadedfiles):
    #             index = i * 6  # Calculate the base index for each file.

    #             poswriter_list = self.all_lists[index]
    #             rotwriter_list = self.all_lists[index + 1]
    #             scalewriter_list = self.all_lists[index + 2]

    #             position_list = self.all_lists[index + 3]
    #             scale_list = self.all_lists[index + 4]
    #             rotation_list = self.all_lists[index + 5]
                
    #             # if self.all_lists[index] is not None:
    #             visible_instances = self.cull_instances(self.loadedfiles[i], base.cam, 20.0, poswriter_list, scalewriter_list, rotwriter_list, position_list, rotation_list, scale_list, max_distance=450)

    #             if visible_instances < 1:
    #                 self.loadedfiles[i].hide()
    #             else:
    #                 self.loadedfiles[i].show()
    #                 self.loadedfiles[i].setInstanceCount(visible_instances)

    #         # time.sleep(0.1)  # Adjust the sleep interval as needed to balance performance

    # def update_camera_lens(self):
    #     # Get the current window size
    #     win_width = base.win.getProperties().getXSize()
    #     win_height = base.win.getProperties().getYSize()
        
    #     # Calculate the new aspect ratio
    #     aspect_ratio = win_width / win_height
        
    #     # Create a new perspective lens and set its aspect ratio
    #     lens = PerspectiveLens()
    #     lens.setAspectRatio(aspect_ratio)
    #     lens.setFov(60)  # You can adjust the field of view if needed
        
    #     # Apply the lens to the camera
    #     base.cam.node().setLens(lens)

    def play_land_sound(self):

        # print(self.player.getActorInfo())
        # print(self.player.get_base_frame_rate())

        # self.walk_sound.setPlayRate(0.6)
        self.land_sound.play()

    def play_walk_sound(self):
        taskMgr.remove("CheckAnimationFrameTask")
        self.run_sound.stop()
        # self.walk_sound.setVolume(0.004 * self.master_volume)
        # # Generate a random play rate between 0.8 and 1.2 for pitch variation
        # # random_play_rate = random.uniform(0.3, 0.4)
        # self.walk_sound.setPlayRate(0.6)
        # self.walk_sound.play()
        taskMgr.add(lambda task: self.check_animation_frame(task, "Walk"), "CheckAnimationFrameTask")

    def stop_walk_sound(self):
        taskMgr.remove("CheckAnimationFrameTask")
        self.walk_sound.stop()
        self.run_sound.stop()
    def play_run_sound(self):
        taskMgr.remove("CheckAnimationFrameTask")
        self.walk_sound.stop()
        self.run_sound.stop()  # Stop any previous sound before starting a new loop
        
        # taskMgr.add(self.check_animation_frame, "CheckAnimationFrameTask")

        # taskMgr.add(self.check_animation_frame, "CheckAnimationFrameTask", extraArgs=[Anim])
        taskMgr.add(lambda task: self.check_animation_frame(task, "Run"), "CheckAnimationFrameTask")

    def check_animation_frame(self, task, current_anim):
        # print(self.player.getCurrentAnim())
        # if self.player.getCurrentAnim() == "Run":
        # frame = self.player.getCurrentFrame("Run")
        
        frame = self.player.getCurrentFrame(current_anim)
        if frame != self.prev_frame:  # Check if the frame has changed
            if frame == 5:  # Left foot
                self.play_footstep_sound(left=True, anim=current_anim)
            elif frame == 15:  # Right foot
                self.play_footstep_sound(left=False, anim=current_anim)
            self.prev_frame = frame
        return Task.cont

    def play_footstep_sound(self, left, anim):
        # Randomize properties
        if anim == "Walk":
            random_play_rate = random.uniform(0.6, 0.7)
            random_volume = 0.004 * self.master_volume
        elif anim == "Run":
            random_play_rate = random.uniform(0.62, 0.85)
            random_volume = 0.005 * self.master_volume
        elif anim == "Sprint":
            random_play_rate = random.uniform(0.76, 0.84)
            random_volume = 0.006 * self.master_volume


        self.run_sound.setPlayRate(random_play_rate)
        self.run_sound.setVolume(random_volume)
        
        # Set the 3D position based on the side
        if left:
            position = LPoint3f(-1, 0, 0)  # Adjust for left side
        else:
            position = LPoint3f(1, 0, 0)  # Adjust for right side
        self.run_sound.set3dAttributes(position.getX(), position.getY(), position.getZ(), 0, 0, 0)

        # Play the sound
        self.run_sound.play()


    def stop_run_sound(self):
        taskMgr.remove("CheckAnimationFrameTask")
        self.run_sound.stop()
        self.actor.stop()

    def play_sprint_sound(self):
        self.run_sound.stop()
        taskMgr.remove("CheckAnimationFrameTask")
        # self.walk_sound.stop()
        # self.run_sound.setVolume(0.011 * self.master_volume)
        # self.run_sound.setPlayRate(0.76)
        # self.run_sound.play()
        taskMgr.add(lambda task: self.check_animation_frame(task, "Sprint"), "CheckAnimationFrameTask")
    def play_dig_sound(self):

        self.dig_sound.setVolume(0.002 * self.master_volume)
        # self.dig_sound.setPlayRate(0.76)
        self.dig_sound.play()
    def play_chop_sound(self):

        self.chop_sound.setVolume(0.002 * self.master_volume)
        # self.dig_sound.setPlayRate(0.76)
        self.chop_sound.play()
    def incremntupfunc(self):

        self.incre=1
    def incremntdownfunc(self):

        self.incre=0
    #placements
    def upfunc(self):

        pos=self.pointer.getPos()
        # Get the heading of the camera
        heading = base.camera.getHpr(render)[0]

        # Normalize the heading to a range of 0 to 360 degrees
        normalized_heading = heading % 360

        # Determine the facing direction based on the heading value and set the new position
        if 45 <= normalized_heading < 135:
            direction = "East"
            self.pointer.setPos(pos[0] - 1 - self.incre, pos[1], pos[2])
        elif 135 <= normalized_heading < 225:
            direction = "South"
            self.pointer.setPos(pos[0], pos[1] - 1 - self.incre, pos[2])
        elif 225 <= normalized_heading < 315:
            direction = "West"
            self.pointer.setPos(pos[0] + 1 + self.incre, pos[1], pos[2])
        else:
            direction = "North"
            self.pointer.setPos(pos[0], pos[1] + 1 + self.incre, pos[2])


        # self.pointer.setPos(pos[0],pos[1]+1+self.incre,pos[2])
    def downfunc(self):

        pos=self.pointer.getPos()
        heading = base.camera.getHpr(render)[0]

        # Normalize the heading to a range of 0 to 360 degrees
        normalized_heading = heading % 360

        # Determine the facing direction based on the heading value and set the new position
        if 45 <= normalized_heading < 135:
            direction = "East"
            self.pointer.setPos(pos[0] + 1 + self.incre, pos[1], pos[2])
        elif 135 <= normalized_heading < 225:
            direction = "South"
            self.pointer.setPos(pos[0], pos[1] + 1 + self.incre, pos[2])
        elif 225 <= normalized_heading < 315:
            direction = "West"
            self.pointer.setPos(pos[0] - 1 - self.incre, pos[1], pos[2])
        else:
            direction = "North"
            self.pointer.setPos(pos[0], pos[1] - 1 - self.incre, pos[2])
        # self.pointer.setPos(pos[0],pos[1]-1-self.incre,pos[2])
    def rightfunc(self):

        pos=self.pointer.getPos()
        # Get the heading of the camera (or player)
        heading = base.camera.getHpr(render)[0]

        # Normalize the heading to a range of 0 to 360 degrees
        normalized_heading = heading % 360

        # Determine the facing direction based on the heading value and set the new position to the right of the facing direction
        if 45 <= normalized_heading < 135:
            direction = "East"
            self.pointer.setPos(pos[0], pos[1] + 1 + self.incre, pos[2])  # Move to the right of East (South)
        elif 135 <= normalized_heading < 225:
            direction = "South"
            self.pointer.setPos(pos[0] - 1 - self.incre, pos[1], pos[2])  # Move to the right of South (West)
        elif 225 <= normalized_heading < 315:
            direction = "West"
            self.pointer.setPos(pos[0], pos[1] - 1 - self.incre, pos[2])  # Move to the right of West (North)
        else:
            direction = "North"
            self.pointer.setPos(pos[0] + 1 + self.incre, pos[1], pos[2])  # Move to the right of North (East)



        # self.pointer.setPos(pos[0]+1+self.incre,pos[1],pos[2])
    def leftfunc(self):

        pos=self.pointer.getPos()
        # Get the heading of the camera (or player)
        heading = base.camera.getHpr(render)[0]

        # Normalize the heading to a range of 0 to 360 degrees
        normalized_heading = heading % 360

        # Determine the facing direction based on the heading value and set the new position to the right of the facing direction
        if 45 <= normalized_heading < 135:
            direction = "East"
            self.pointer.setPos(pos[0], pos[1] - 1 - self.incre, pos[2])  # Move to the right of East (South)
        elif 135 <= normalized_heading < 225:
            direction = "South"
            self.pointer.setPos(pos[0] + 1 + self.incre, pos[1], pos[2])  # Move to the right of South (West)
        elif 225 <= normalized_heading < 315:
            direction = "West"
            self.pointer.setPos(pos[0], pos[1] + 1 + self.incre, pos[2])  # Move to the right of West (North)
        else:
            direction = "North"
            self.pointer.setPos(pos[0] - 1 - self.incre, pos[1], pos[2])  # Move to the right of North (East)
        # self.pointer.setPos(pos[0]-1-self.incre,pos[1],pos[2])
    def zupfunc(self):

        pos=self.pointer.getPos()
        self.zmem+=1
        self.pointer.setPos(pos[0],pos[1],pos[2]+1+self.incre)
    def zdownfunc(self):

        pos=self.pointer.getPos()
        self.zmem-=1
        self.pointer.setPos(pos[0],pos[1],pos[2]-1-self.incre)
    def placefunc(self):

        # ib=5
        index = self.modelnames.index(self.itemdex)

        ppos=self.pointer.getPos()
        phpr=self.pointer.getHpr()
        # Get the transformation matrix of self.pointer relative to the root node
        # mat = self.pointer.getMat(self.render)
        # position = Point3(mat.getRow3(3))


        hmap0=(int(ppos[0]  // 486) * 486,int(ppos[1] // 486) * 486)
        heightmapsgot = self.height_maps.get((hmap0))

        self.setup_instance(heightmapsgot,self.loadedfiles[index],[ppos[0],ppos[1],ppos[2]],hprb=[phpr[0],phpr[1],phpr[2]], poswriter=self.all_lists[index*6],rotwriter=self.all_lists[index*6+1],scalewriter=self.all_lists[index*6+2], zrotonly=False, scaler=[1,1],indexB=0,names=self.modelnames[index])
    def resetrotfunc(self):

        self.zmem=0
        self.pointer.setHpr(0,0,0)
        self.rotationsmem=[0,0,0]
    def rotforwardfunc(self):


        rot=self.pointer.getHpr()
        heading = base.camera.getHpr(render)[0]

        # Normalize the heading to a range of 0 to 360 degrees
        normalized_heading = heading % 360

        # Determine the facing direction based on the heading value and set the new position to the right of the facing direction
        if 45 <= normalized_heading < 135:
            direction = "East"
            self.rotationsmem[2]-=45
            self.pointer.setHpr(rot[0],rot[1],rot[2]-45)
            # self.pointer.setPos(pos[0], pos[1] - 1 - self.incre, pos[2])  # Move to the right of East (South)
        elif 135 <= normalized_heading < 225:
            direction = "South"
            self.rotationsmem[1]+=45
            self.pointer.setHpr(rot[0],rot[1]+45,rot[2])
            # self.pointer.setPos(pos[0] + 1 + self.incre, pos[1], pos[2])  # Move to the right of South (West)
        elif 225 <= normalized_heading < 315:
            direction = "West"
            self.rotationsmem[2]+=45
            self.pointer.setHpr(rot[0],rot[1],rot[2]+45)
            # self.pointer.setPos(pos[0], pos[1] + 1 + self.incre, pos[2])  # Move to the right of West (North)
        else:
            direction = "North"
            self.rotationsmem[1]-=45
            self.pointer.setHpr(rot[0],rot[1]-45,rot[2])

            # self.pointer.setPos(pos[0] - 1 - self.incre, pos[1], pos[2])  # Move to the right of North (East)
    def rotdownwardfunc(self):

        rot=self.pointer.getHpr()

        heading = base.camera.getHpr(render)[0]

        # Normalize the heading to a range of 0 to 360 degrees
        normalized_heading = heading % 360

        # Determine the facing direction based on the heading value and set the new position to the right of the facing direction
        if 45 <= normalized_heading < 135:
            direction = "East"
            self.rotationsmem[2]+=45
            self.pointer.setHpr(rot[0],rot[1],rot[2]+45)
            # self.pointer.setPos(pos[0], pos[1] - 1 - self.incre, pos[2])  # Move to the right of East (South)
        elif 135 <= normalized_heading < 225:
            direction = "South"
            self.rotationsmem[1]-=45
            self.pointer.setHpr(rot[0],rot[1]-45,rot[2])
            # self.pointer.setPos(pos[0] + 1 + self.incre, pos[1], pos[2])  # Move to the right of South (West)
        elif 225 <= normalized_heading < 315:
            direction = "West"
            self.rotationsmem[2]-=45
            self.pointer.setHpr(rot[0],rot[1],rot[2]-45)
            # self.pointer.setPos(pos[0], pos[1] + 1 + self.incre, pos[2])  # Move to the right of West (North)
        else:
            direction = "North"
            self.rotationsmem[1]+=45
            self.pointer.setHpr(rot[0],rot[1]+45,rot[2])

        # self.rotationsmem[1]+=45
        # self.pointer.setHpr(rot[0],rot[1]+45,rot[2])
    def rotleftfunc(self):

        rot=self.pointer.getHpr()

        heading = base.camera.getHpr(render)[0]

        # Normalize the heading to a range of 0 to 360 degrees
        normalized_heading = heading % 360

        # Determine the facing direction based on the heading value and set the new position to the right of the facing direction
        if 45 <= normalized_heading < 135:
            direction = "East"
            self.rotationsmem[1]+=45
            self.pointer.setHpr(rot[0],rot[1]+45,rot[2])
            # self.pointer.setPos(pos[0], pos[1] - 1 - self.incre, pos[2])  # Move to the right of East (South)
        elif 135 <= normalized_heading < 225:
            direction = "South"
            self.rotationsmem[2]+=45
            self.pointer.setHpr(rot[0],rot[1],rot[2]+45)
            # self.pointer.setPos(pos[0] + 1 + self.incre, pos[1], pos[2])  # Move to the right of South (West)
        elif 225 <= normalized_heading < 315:
            direction = "West"
            self.rotationsmem[1]-=45
            self.pointer.setHpr(rot[0],rot[1]-45,rot[2])
            # self.pointer.setPos(pos[0], pos[1] + 1 + self.incre, pos[2])  # Move to the right of West (North)
        else:
            direction = "North"
            self.rotationsmem[2]-=45
            self.pointer.setHpr(rot[0],rot[1],rot[2]-45)

        # self.rotationsmem[2]-=45
        # self.pointer.setHpr(rot[0],rot[1],rot[2]-45)
    def rotrightfunc(self):

        rot=self.pointer.getHpr()

        heading = base.camera.getHpr(render)[0]

        # Normalize the heading to a range of 0 to 360 degrees
        normalized_heading = heading % 360

        # Determine the facing direction based on the heading value and set the new position to the right of the facing direction
        if 45 <= normalized_heading < 135:
            direction = "East"
            self.rotationsmem[1]-=45
            self.pointer.setHpr(rot[0],rot[1]-45,rot[2])
            # self.pointer.setPos(pos[0], pos[1] - 1 - self.incre, pos[2])  # Move to the right of East (South)
        elif 135 <= normalized_heading < 225:
            direction = "South"
            self.rotationsmem[2]-=45
            self.pointer.setHpr(rot[0],rot[1],rot[2]-45)
            # self.pointer.setPos(pos[0] + 1 + self.incre, pos[1], pos[2])  # Move to the right of South (West)
        elif 225 <= normalized_heading < 315:
            direction = "West"
            self.rotationsmem[1]+=45
            self.pointer.setHpr(rot[0],rot[1]+45,rot[2])
            # self.pointer.setPos(pos[0], pos[1] + 1 + self.incre, pos[2])  # Move to the right of West (North)
        else:
            direction = "North"
            self.rotationsmem[2]+=45
            self.pointer.setHpr(rot[0],rot[1],rot[2]+45)

        # self.rotationsmem[2]+=45
        # self.pointer.setHpr(rot[0],rot[1],rot[2]+45)
    def rotheadingleftfunc(self):

        rot=self.pointer.getHpr()
        self.rotationsmem[0]-=45
        self.pointer.setHpr(rot[0]-45,rot[1],rot[2])
    def rotheadingrightfunc(self):

        rot=self.pointer.getHpr()
        self.rotationsmem[0]+=45
        self.pointer.setHpr(rot[0]+45,rot[1],rot[2])
    def placelogfunc(self):

        self.itemdex = "log0_0_0_0_1_2"
        if self.todisplay:
            self.todisplay.hide()
        self.todisplay = None

        self.update_display_item()

    def placepanelfunc(self):

        self.itemdex = "panel0_0_0_0_1_2"
        if self.todisplay:
            self.todisplay.hide()
        self.todisplay = None
        self.update_display_item()

    def placestickfunc(self):

        self.itemdex = "stick0_0_0_0_1_2"
        if self.todisplay:
            self.todisplay.hide()
        self.todisplay = None
        self.update_display_item()

    def key_graphics(self):

        self.pgraphics_frame.show()
        self.pause_frame.hide()
        # if self.keytog_graphics:

        # self.keytog_graphics = not self.keytog_graphics


    def setfovfunc(self):
        int_value = round(self.slider['value'])
        self.fovtext.setText(f'FOV {int_value}')
        base.camLens.setFov(int_value)
    def setvolfunc(self):
        int_value = round(self.sliderVolume['value'])
        self.volumetext.setText(f'Volume {int_value}')
        self.set_master_volume(int_value)
        # base.camLens.setFov(int_value)

    def change_to_fullscreen(self):
        wp = WindowProperties()
        wp.setFullscreen(True)
        wp.setSize(1920, 1080)
        base.win.requestProperties(wp)

    def change_window_size(self, new_width=1280, new_height=720, fovv=90):
        props = WindowProperties()
        props.setSize(new_width, new_height)
        base.win.requestProperties(props)

        # Update the aspect ratio of the camera's lens
        aspect_ratio = new_width / new_height
        base.camLens.setAspectRatio(aspect_ratio)
        base.camLens.setFov(fovv)
        
    def apply(self):
        if self.keypointhidetog == False:
            self.data_loaded["pointer"]=0
        else:
            self.data_loaded["pointer"]=1
        if self.keynormalstog == False:
            self.data_loaded["align"]=0
        else:
            self.data_loaded["align"]=1
        int_value = round(self.slider['value'])
        self.data_loaded["fov"]=int_value
        int_value = round(self.sliderVolume['value'])
        self.data_loaded["volume"]=int_value
        with open(self.filename, "w") as file:
            json.dump(self.data_loaded, file, indent=4)


    def on_window_event(self, window):
        if window is not None:
            self.player.update_window_size()
            props = window.getProperties()
            current_wpb0 = self.win.getProperties()
            
            # Get the current window size
            current_widthb0 = current_wpb0.getXSize()
            current_heightb0 = current_wpb0.getYSize()
            scaleb0 = (8*120.0 / current_widthb0, 1, 8*80.0 / current_heightb0)

            self.gui_root.setScale(scaleb0)

            if props.getForeground():

                # Add your resume game logic here
                if self.fs == True:
                    self.fs=False
                    self.pause=False
                    self.pause_frame.hide()

            else:

                if self.pause == False:
                    self.togglePause()

    def resume_game(self):
        if self.pause == True:
            self.togglePause()

    def pause(self):
        if not self.pause:
            self.togglePause()
    def pauseb(self):
        if not self.pause:
            self.hidePause()

    def hidePause(self):
        # if self.pause and self.pauseb:
        #     self.pause_frame.hide()
        #     self.onscreengui_frame.hide()
        # else:
        if self.pauseb:
            self.pause_frame.hide()
            self.onscreengui_frame.hide()
            # self.holding.hide()
            # self.pgraphics_frame.hide()
        else:
            if self.pause == True:
                self.pause_frame.show()
            self.onscreengui_frame.show()
            # self.holding.show()
        self.pauseb = not self.pauseb

    def togglePause(self):
        """This function shows how the app can pause and resume the
        player"""
        if not self.interact_frame.isHidden():
            # self.interact_frame.hide()
            # self.player.resumePlayer()
            # props = WindowProperties()
            # props.setCursorHidden(True)
            # base.win.requestProperties(props)
            if self.keyCtog == True:
                self.keyC()
        else:
            if self.pause:
                # self.pauseb=False
                # if self.pauseb == False:
                #     self.hidePause() 
                self.player.resumePlayer()
                # self.resume_button.hide()
                self.pgraphics_frame.hide()
                self.pause_frame.hide()
                # self.res_button.hide()
                props = WindowProperties()
                props.setCursorHidden(True)
                base.win.requestProperties(props)
            else:
                if self.key1tog == True:
                    self.key1()
                if self.key2tog == True:
                    self.key2()
                # if self.key3tog == True:
                #     self.key3()
                if self.key4tog == True:
                    self.key4()
                self.holding['text']=''
                self.pointer.hide()
                if self.pauseb == False:
                    self.hidePause() 
                # self.pauseb=True
                # if self.pauseb == False:
                #     self.hidePause() 
                self.player.pausePlayer()
                self.player.flymodeon()
                # self.resume_button.show()
                self.pause_frame.show()
                self.walk_sound.stop()
                self.run_sound.stop()
                taskMgr.remove("CheckAnimationFrameTask")
                if self.keyCtog == True:
                    self.keyC()
                # Show mouse cursor
                props = WindowProperties()
                props.setCursorHidden(False)
                base.win.requestProperties(props)
                # self.res_button.show()
            self.pause = not self.pause

    def on_close_request(self):
        # sys.exit()
        self.quit()

    def key1tog(self):
        if not self.key1tog:
            self.key1()      
            
    def key1(self):
        if not self.pauseb == True:
            self.key1tog = not self.key1tog
        else:
            self.key1tog=False

        if self.key2tog == True:
            self.key2()
        # if self.key3tog == True:
        #     self.key3()
        if self.key4tog == True:
            self.key4()
        if self.key1tog == True:
            self.holding['text']='1'
            self.pointer.hide()
        else:
            self.holding['text']=''

    def key2tog(self):
        if not self.key2tog:
            self.key2()
            
    def key2(self):
        if not self.pauseb == True:
            self.key2tog = not self.key2tog
        else:
            self.key2tog=False

        if self.key1tog == True:
            self.key1()
        # if self.key3tog == True:
        #     self.key3()
        if self.key4tog == True:
            self.key4()
        self.clicked = False
        if self.key2tog == True:
            self.holding['text']='2'
        else:
            self.holding['text']=''
            self.pointer.hide()
            if self.keyCtog == True:
                self.keyC()

    def keyCtog(self):
        if not self.keyCtog:
            self.keyC()

    def keyC(self):
        if not self.pauseb == True:
            self.keyCtog = not self.keyCtog
        else:
            self.keyCtog=False

        if self.keyCtog == True:
            self.interact_frame.show()
            self.player.pausePlayer()
            props = WindowProperties()
            props.setCursorHidden(False)
            base.win.requestProperties(props)
            self.walk_sound.stop()
            self.run_sound.stop()
            taskMgr.remove("CheckAnimationFrameTask")
            if self.key2tog == False:
                self.key2()
        else:
            self.interact_frame.hide()
            self.player.resumePlayer()
            props = WindowProperties()
            props.setCursorHidden(True)
            base.win.requestProperties(props)

    def keypointhidetog(self):
        if not self.keypointhidetog:
            self.keypointhide()

    def keypointhide(self):
        self.keypointhidetog = not self.keypointhidetog
        if self.keypointhidetog == True:
            self.pointer.show()
            self.rshowpointer_button['text'] = 'pointer: on'
        else:
            self.pointer.hide()
            self.rshowpointer_button['text'] = 'pointer: off'

    def keynormalstog(self):
        if not self.keynormalstog:
            self.keynormals()

    def keynormals(self):
        self.keynormalstog = not self.keynormalstog
        if self.keynormalstog == True:
            self.ralighcursor_button ['text'] = 'align: on'
        else:
            self.pointer.setHpr(0,0,0)
            self.ralighcursor_button ['text'] = 'align: off'

    def key3tog(self):
        if not self.key3tog:
            self.key3()
            
    def key3(self):
        # if not self.pauseb == True:
        self.key3tog = not self.key3tog
        # else:
        #     self.key3tog=False
        if self.key3tog==True:
            self.render.set_shader_input(f'spot_lights[0].color', LVecBase4(1, 1, 1, 1))#on
            self.render.set_shader_input('point_lights[0].color', LVecBase4(1, 1, 1, 1))
        else:
            self.render.set_shader_input(f'spot_lights[0].color', LVecBase4(0, 0, 0, 0))#off
            self.render.set_shader_input('point_lights[0].color', LVecBase4(0, 0, 0, 0))

        if self.key1tog == True:
            self.key1()
        if self.key2tog == True:
            self.key2()
        if self.key4tog == True:
            self.key4()

    def key4tog(self):
        if not self.key4tog:
            self.key4()
            
    def key4(self):
        if not self.pauseb == True:
            self.key4tog = not self.key4tog
        else:
            self.key4tog=False

        if self.key1tog == True:
            self.key1()
        if self.key2tog == True:
            self.key2()
        if self.key3tog == True:
            self.key3()

    def quit(self):
        # Example of updating the position
        # new_position = [10, 20, 30]
        plhprget=self.player.plugin_getHpr()
        plposget=self.player.plugin_getPos()
        self.world_data_player["position"] = [plposget[0],plposget[1],plposget[2]]
        self.world_data_player["facing"] = [plhprget[0],plhprget[1],plhprget[2]]
        self.world_data_player["time"][0]=self.custom_time % 360
        self.world_data_player["time"][1]=self.moon_phase_counter

        # Write the updated data back to the JSON file
        with open(self.filename_world, "w") as file:
            json.dump(self.world_data_player, file, indent=4)

        json_object = json.dumps(self.savesedit)
        # Write the JSON object to the file
        world_name_n = "saves/"+self.worldname+"/world.json"
        with open(world_name_n, "w") as outfile:
            outfile.write(json_object)

        json_object = json.dumps(self.forplacement)
        # Write the JSON object to the file
        world_name_p = "saves/"+self.worldname+"/placedata.json"
        with open(world_name_p, "w") as outfile:
            outfile.write(json_object)

        json_object = json.dumps(self.forremoval)
        # Write the JSON object to the file
        world_name_r = "saves/"+self.worldname+"/removeddata.json"
        with open(world_name_r, "w") as outfile:
            outfile.write(json_object)
        sys.exit()

    def update_display_item(self):
        if self.todisplay is not None:
            if self.todisplay.isHidden():
                self.todisplay.show()
            self.todisplay.setPos(self.pointer.getPos(render))  # Using global position
            self.todisplay.setHpr(self.pointer.getHpr(render))  # Using global orientation
            self.todisplay.reparentTo(self.pointer)
        else:
            index = self.modelnames.index(self.itemdex)
            self.todisplay, _ = self.displaymodels[index]
            self.todisplay.show()
            self.todisplay.setPos(self.pointer.getPos(render))  # Using global position
            self.todisplay.setHpr(self.pointer.getHpr(render))  # Using global orientation
            self.todisplay.reparentTo(self.pointer)
        
        # Apply CompassEffect to exclude scale inheritance
        compass_effect = CompassEffect.make(render, CompassEffect.PScale)
        self.todisplay.node().setEffect(compass_effect)

        # Reset position and orientation relative to the new parent
        self.todisplay.setPos(0, 0, 0)
        self.todisplay.setHpr(0, 0, 0)
        self.todisplay.setTransparency(TransparencyAttrib.MAlpha)
        self.todisplay.setColor(1, 1, 1, 0.5)  # RGBA: Red, Green, Blue, Alpha

    # Method to search for a model by name in self.loadedfiles
    def search_loaded_files(self, model_name):
        for model, collider in self.loadedfiles:
            if model.getName() == model_name:
                return model, collider
        return None, None

    def load_bam_filesb(self, loader, directory, shader, alternate_shader):
        for root, dirs, files in os.walk(directory):
            folder_name = os.path.basename(root)
            loadedfiles = []
            loadedfiles2 = []
            modelnames = []

            for file in files:
                if file.endswith('.bam'):
                    model_name = os.path.splitext(file)[0]

                    # Load the original model
                    model = loader.loadModel(os.path.join(root, file))
                    model.setName(model_name)

                    model.setScale(1)
                    model.setShader(shader)
                    model.reparentTo(render)
                    model.setTwoSided(True)
                    model.setTransparency(TransparencyAttrib.MAlpha)
                    model.set_bin('opaque', 1)

                    # Load the duplicate model
                    duplicate_model = loader.loadModel(os.path.join(root, file))
                    duplicate_model.setName(model_name + "_duplicate")

                    duplicate_model.setScale(1)
                    duplicate_model.setShader(alternate_shader)
                    duplicate_model.reparentTo(render)
                    duplicate_model.set_transparency(TransparencyAttrib.M_alpha)
                    duplicate_model.setShaderInput("modelColor", Vec4(1, 1, 1, 0.5))
                    duplicate_model.set_attrib(DepthWriteAttrib.make(DepthWriteAttrib.M_off))
                    duplicate_model.set_bin('transparent', 2)

                    # Update texture paths if necessary
                    colliderb = duplicate_model.find("**/collider")
                    if not colliderb.isEmpty():
                        colliderb.removeNode()

                    collider = model.find("**/collider")
                    if collider.isEmpty():
                        collider = None
                    else:
                        collider.detachNode()
                        collider.hide()

                    tex = model.findTexture("*")
                    parts = model_name.split('_')
                    texture_path = os.path.join(root, parts[0] + '.png')
                    if os.path.exists(texture_path):
                        tex.read(texture_path)
                    else:
                        print(f"Texture file not found: {texture_path}")

                    loadedfiles.append([model, collider])
                    modelnames.append(model_name)
                    duplicate_model.hide()
                    loadedfiles2.append([duplicate_model, collider])

            self.model_data[folder_name] = { 'loadedfiles': loadedfiles, 'modelnames': modelnames, 'loadedfiles2': loadedfiles2 }


        return self.model_data



    def load_bam_files(self, loader, directory, shader, alternate_shader):
        loadedfiles = []
        loadedfiles2 = []
        modelnames = []
        modeldirectory = []
        
        for root, dirs, files in os.walk(directory):

            for file in files:
                if file.endswith('.bam'):
                    model_name = os.path.splitext(file)[0]
                    end_part = os.path.basename(root)

                    # Load the original model
                    model = loader.loadModel(os.path.join(root, file))
                    model.setName(model_name)

                    model.setScale(1)
                    model.setShader(shader)
                    model.reparentTo(render)
                    model.setTwoSided(True)
                    model.setTransparency(TransparencyAttrib.MAlpha)
                    model.set_bin('opaque', 1)

                    # Load the duplicate model
                    duplicate_model = loader.loadModel(os.path.join(root, file))
                    duplicate_model.setName(model_name + "_duplicate")

                    duplicate_model.setScale(1)
                    duplicate_model.setShader(alternate_shader)
                    duplicate_model.reparentTo(render)

                    duplicate_model.set_transparency(TransparencyAttrib.M_alpha)
                    duplicate_model.setShaderInput("modelColor", Vec4(1, 1, 1, 0.5))
                    duplicate_model.set_attrib(DepthWriteAttrib.make(DepthWriteAttrib.M_off))
                    duplicate_model.set_bin('transparent', 2)  # Transparent object

                    # Update texture paths if necessary
                    colliderb = duplicate_model.find("**/collider")

                    if not colliderb.isEmpty(): 
                        colliderb.removeNode()

                    # Update texture paths if necessary
                    collider = model.find("**/collider")

                    if collider.isEmpty():
                        collider = None
                    else:
                        collider.detachNode()
                        collider.hide()

                    tex = model.findTexture("*")
                    parts = model_name.split('_')

                    texture_path = os.path.join(root, parts[0] + '.png')
                    if os.path.exists(texture_path):
                        tex.read(texture_path)
                    else:
                        print(f"Texture file not found: {texture_path}")

                    loadedfiles.append([model, collider])
                    modelnames.append(model_name)
                    duplicate_model.hide()

                    loadedfiles2.append([duplicate_model, collider])
                    modeldirectory.append(end_part)

        return loadedfiles, modelnames, loadedfiles2, modeldirectory

    def genter(self, task):
        # if self.distance_moved > 5:

        for x_check in range(self.prev_x, self.x_positive, 54):
            for z_check in range(self.prev_z, self.z_positive, 54):
                self.generate_terrain_if_needed(x_check, z_check)

        for x_check in range(self.prev_x, self.x_negative, -54):
            for z_check in range(self.prev_z, self.z_negative, -54):
                self.generate_terrain_if_needed(x_check, z_check)

        return task.again# Indicate that the task is complete

    def removeradius(self, task):
        self.remove_far_meshes0(200)    
        self.remove_far_meshes1(972) 
        return task.done # Indicate that the task is complete

    def cull_and_update_instances(self, node, camera, radius, poswriter, scalewriter, rotwriter, max_distance=50):
        lens = camera.node().getLens()
        lensBounds = lens.makeBounds()
        visible_instances = 0
        # self.countbb += 1
        if poswriter is not None:
            poswriter.setRow(0)
            rotwriter.setRow(0)
            scalewriter.setRow(0)
            camera_pos = camera.getPos(node)
            bounds = BoundingSphere()
            node_mat = node.getMat(camera)  # Cache the node's transformation
            camera_dir = camera.get_quat(render).get_forward()
            distance = 300
            directional_vector = camera_dir * distance
            new_position = camera_pos + directional_vector
            base_x_key = int(new_position.x // 162) * 162
            base_z_key = int(new_position.y // 162) * 162
            offsets = [-324, -162, 0, 162, 324]
            for offset_x in offsets:
                for offset_z in offsets:
                    new_x_key = base_x_key + offset_x
                    new_z_key = base_z_key + offset_z
                    diremp = (new_x_key, new_z_key)
                    combined_key = f"{node.getName()}{diremp}"
                    instances_pos = self.instance_groups.get(combined_key)
                    if instances_pos is not None:
                        for pos, rotation, scale in zip(instances_pos[0], instances_pos[1], instances_pos[2]):
                            instance_position = LPoint3f(*pos)
                            distance_from_camera_sq = (instance_position - camera_pos).lengthSquared()
                            bounds.setCenter(instance_position)
                            bounds.setRadius(radius)
                            bounds.xform(node_mat)
                            if lensBounds.contains(bounds) or (distance_from_camera_sq >= (30**2)) and distance_from_camera_sq <= (max_distance**2):
                                poswriter.add_data3(instance_position)
                                rotwriter.add_data4(LVecBase4f(rotation[0], rotation[1], rotation[2], 1))
                                scalewriter.add_data4(LVecBase4f(scale[0], scale[1], scale[2], 1))
                                visible_instances += 1

        return visible_instances

    def choose_random_folder(self, parent_folder):
        # Get a list of all folders in the parent folder
        all_folders = [f for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, f))]
        
        # Choose a random folder
        random_folder = random.choice(all_folders)
        
        return random_folder

    def choose_random_files(self, folder_path, num_files):
        # Get a list of all .npy files in the folder
        npy_files = [f for f in os.listdir(folder_path) if f.endswith('map.png') and os.path.isfile(os.path.join(folder_path, f))]
        # Choose random .npy files
        random_files = random.sample(npy_files, num_files)
        
        return random_files


    def load_specific_files(self, folder_path, file_names, buffer):
        loaded_files = []
        # if folder_path == "data/level/terrains/desert":
            # print(folder_path, file_names)
            # print(buffer)
        for file_name in file_names:
            buffered_file = next((item for item in buffer if item[3] == file_name), None)
            # print(file_name)
            if buffered_file:
                # print(len(buffered_file),buffer)
                # If the file is already in the buffer, append it to loaded_files

                loaded_files.append(buffered_file)
            else:
                try:
                    file_path_png = os.path.join(folder_path, file_name + '.png')
                    file_path_map_png = os.path.join(folder_path, file_name + 'map.png')
                    file_path_region_png = os.path.join(folder_path, file_name + 'region.png')

                    brush_height2region = PNMImage()
                    brush_height2region.read(Filename(file_path_region_png))
                    widthreg = brush_height2region.get_x_size()
                    heightreg = brush_height2region.get_y_size()
                    brush_heightregion = np.array([[brush_height2region.get_gray(x, y) * 255 for x in range(widthreg)] for y in range(heightreg)], dtype=np.uint8)

                    brush_height2im = PNMImage()
                    brush_height2im.read(Filename(file_path_map_png))
                    
                    width = brush_height2im.get_x_size()
                    height = brush_height2im.get_y_size()
                    
                    brush_height = np.array([[brush_height2im.get_gray(x, y) * 255 for x in range(width)] for y in range(height)], dtype=np.uint8)

                    bigtest = PNMImage()
                    bigtest.read(Filename(file_path_png))
                    
                    normalized_path = os.path.normpath(folder_path)
                    end_folder = os.path.basename(normalized_path)
                    
                    loaded_file = (brush_height, bigtest, end_folder, file_name, brush_heightregion)
                    loaded_files.append(loaded_file)
                    
                    # Add the file to the buffer
                    buffer.append(loaded_file)
                except Exception as e:
                    print(f"Error loading files: {e}")
        
        return loaded_files


    def load_random_file(self, folder):
        files = [f for f in os.listdir(folder) if f.endswith('.png') and not f.endswith('map.png') and not f.endswith('region.png')]
        if not files:
            raise FileNotFoundError(f"No PNG files found in {folder}")
        chosen_file = random.choice(files)
        return os.path.splitext(chosen_file)[0]

    def get_terrain_file(self, x, y, parent_folder):
        def calculate_probability(dx, dy, radius=100):
            return max(0, 1 - math.sqrt(dx**2 + dy**2) / radius)
        
        biomes = {
            'mountains': calculate_probability(x, y),
            'desert': calculate_probability(x-1000, y-1000, radius=1500)  # Increase the radius here
        }
        
        total_prob = sum(biomes.values())
        if total_prob == 0:
            folder = 'mountains/'
        else:
            r = random.random() * total_prob
            cumulative_prob = 0.0
            for biome, prob in biomes.items():
                cumulative_prob += prob
                if r < cumulative_prob:
                    folder = biome + '/'
                    break
        
        full_path = os.path.join(parent_folder, 'terrains', folder)
        truebiome = folder.strip('/')

        return truebiome, self.load_random_file(full_path)

    def painter(self, base_height, brush_height, offset_x=0, offset_z=0, invert=False):
        # Calculate the indices for slicing as before
        start_x = max(offset_x, 0)
        end_x = min(base_height.shape[1], offset_x + brush_height.shape[1])
        start_z = max(offset_z, 0)
        end_z = min(base_height.shape[0], offset_z + brush_height.shape[0])

        brush_start_x = max(-offset_x, 0)
        brush_end_x = min(brush_height.shape[1], brush_start_x + end_x - start_x)

        brush_start_z = max(-offset_z, 0)
        brush_end_z = min(brush_height.shape[0], brush_start_z + end_z - start_z)

        if invert:
            base_height[start_z:end_z, start_x:end_x] -= brush_height[brush_start_z:brush_end_z, brush_start_x:brush_end_x]
        else:
            base_height[start_z:end_z, start_x:end_x] += brush_height[brush_start_z:brush_end_z, brush_start_x:brush_end_x]

        return base_height

    def painterreplace(self, base_height, brush_height, offset_x=0, offset_z=0, invert=False):
        # Calculate the indices for slicing
        start_x = max(offset_x, 0)
        end_x = min(base_height.shape[1], offset_x + brush_height.shape[1])
        start_z = max(offset_z, 0)
        end_z = min(base_height.shape[0], offset_z + brush_height.shape[0])

        brush_start_x = max(-offset_x, 0)
        brush_end_x = min(brush_height.shape[1], brush_start_x + end_x - start_x)

        brush_start_z = max(-offset_z, 0)
        brush_end_z = min(brush_height.shape[0], brush_start_z + end_z - start_z)

        # Slice the relevant portions of base_height and brush_height
        base_slice = base_height[start_z:end_z, start_x:end_x]
        brush_slice = brush_height[brush_start_z:brush_end_z, brush_start_x:brush_end_x]

        # Apply the condition to replace values only if they are greater than zero
        mask = brush_slice > 0
        if invert:
            base_slice[mask] -= brush_slice[mask]
        else:
            base_slice[mask] = brush_slice[mask]
        return base_height

    def calculate_normal(self, x, y, height_map):
        # Get the height of the neighbors or the vertex itself if it's an edge
        height_x_plus_1 = height_map[x + 1][y] if x < len(height_map) - 1 else height_map[x][y]
        height_x_minus_1 = height_map[x - 1][y] if x > 0 else height_map[x][y]
        height_y_plus_1 = height_map[x][y + 1] if y < len(height_map[0]) - 1 else height_map[x][y]
        height_y_minus_1 = height_map[x][y - 1] if y > 0 else height_map[x][y]

        dx = (height_x_plus_1 - height_x_minus_1) / 2.0
        dy = (height_y_plus_1 - height_y_minus_1) / 2.0
        normal = Vec3(-dx, -dy, 1).normalized()
        return normal

    def position_gen(self, value=1000, seed=29, area_width=486, area_height=486, height_map=None):
        random.seed(seed)
        occupied_positions = []

        for i in range(value):
            while True:
                x = random.uniform(0, area_width)
                y = random.uniform(0, area_height)
                z = height_map[int(x)][int(y)]
                pos = (int(x), int(y), z)
                if pos not in occupied_positions:
                    occupied_positions.append(pos)
                    break

        return occupied_positions

    def normal_to_euler(self, normal):
        # Normalize the normal vector
        normal = normal / np.linalg.norm(normal)
        
        # Calculate the Euler angles (Heading, Pitch, Roll)
        pitch = np.arcsin(normal[1])
        heading = -np.arctan2(normal[0], normal[2])  # Invert the heading
        
        # Calculate the roll based on the heading and pitch
        R = np.array([
            [np.cos(heading), -np.sin(heading), 0],
            [np.sin(heading), np.cos(heading), 0],
            [0, 0, 1]
        ])
        
        forward = np.dot(R, np.array([0, 1, 0]))
        roll = np.arccos(np.dot(forward, normal) / np.linalg.norm(normal))
        
        return heading, pitch, roll

    def setup_instancingcloud(self, heightmapsgotfunc, nodefunc, positionlist=[], seed=29, fromvalue=0, tovalue=250, new_location=(0, 0, 0), zlevel=True, total_instances=0, poswriter=None, rotwriter=None, scalewriter=None, zrotonly=False):
        random.seed(seed)
        gnode = nodefunc.find("**/+GeomNode").node()
        offsetpositionlist=[]
        scalelist=[]
        rotationlist=[]
        vdata = gnode.modifyGeom(0).modifyVertexData()
        format = GeomVertexFormat(gnode.getGeom(0).getVertexData().getFormat())
        if not format.hasColumn("offset"):
            iformat = GeomVertexArrayFormat()
            iformat.setDivisor(1)
            iformat.addColumn("offset", 4, Geom.NT_stdfloat, Geom.C_other)
            iformat.addColumn("rotation", 4, Geom.NT_stdfloat, Geom.C_other)
            iformat.addColumn("scale", 4, Geom.NT_stdfloat, Geom.C_other)
            
            format.addArray(iformat)
            format = GeomVertexFormat.registerFormat(format)
            vdata.setFormat(format)

        sorted_positions = positionlist if zlevel is None else sorted(positionlist, key=lambda pos: pos[2], reverse=zlevel)
        
        if poswriter is None:
            poswriter = GeomVertexWriter(vdata.modifyArray(2), 0)
        # if rotwriter is None:
            rotwriter = GeomVertexWriter(vdata.modifyArray(2), 1)  # Writer for rotation
        # if scalewriter is None:
            scalewriter = GeomVertexWriter(vdata.modifyArray(2), 2)  # Writer for scale

        self.instance_parent1 = NodePath('instances')
        for i in range(fromvalue, tovalue):
            placeholder = self.instance_parent1.attachNewNode(f'Instance_{i}')
            x, y, z = sorted_positions[i]
            zrand = random.uniform(2.9, 4.5)
            placeholder.setPos(x, y, z+zrand)
            poswriter.add_data3(x + new_location[0], y + new_location[1], z+ zrand + new_location[2])
            offsetpositionlist.append([x + new_location[0], y + new_location[1], z + zrand + new_location[2]])

            random_heading = random.uniform(0, 10)
            # instance = self.tree.instanceTo(placeholder)
            # instance.setTag('instance', 'true')
            rotwriter.add_data4(random_heading, 0, 0, 1)
            # rotwriter.add_data4(h, p, r, 1)
            scale = random.uniform(2.9, 9.5)
            placeholder.setScale(scale)
            scalewriter.add_data4(scale, scale, scale, 1)
            rotationlist.append([random_heading, 0, 0, 1])
            scalelist.append([scale, scale, scale, 1])

        # Move the whole area to the new location
        self.instance_parent1.setPos(new_location)

        self.instance_parent1.reparentTo(self.render)
        total_instances += (tovalue - fromvalue)
        nodefunc.setInstanceCount(total_instances)
        nodefunc.node().setBounds(OmniBoundingVolume())
        nodefunc.node().setFinal(True)

        return total_instances, poswriter, rotwriter, scalewriter,offsetpositionlist,scalelist,rotationlist


    def setup_instancing(self, heightmapsgotfunc, nodefunc, positionlist=[], seed=29, new_location=(0, 0, 0), zlevel=True, total_instances=0, poswriter=None, rotwriter=None, scalewriter=None, indexB=0, names='',keycord=[0,0],directoyp=''):
        random.seed(seed)
        regionmapsgot = self.region_maps.get(tuple(keycord))

        nodefunc, nodecoll = nodefunc
        parts = str(nodefunc.getName()).split('_')
        value0 = int(parts[1])
        value1 = int(parts[2])
        value2b = int(parts[3])
        if value2b == 1:
            roto=False
        else:
            roto=True
        value3s = int(parts[4])
        value4s = int(parts[5])
        scales=[value3s,value4s]
        gnode = nodefunc.find("**/+GeomNode").node()
        vdata = gnode.modifyGeom(0).modifyVertexData()
        format = GeomVertexFormat(gnode.getGeom(0).getVertexData().getFormat())
        if not format.hasColumn("offset"):
            iformat = GeomVertexArrayFormat()
            iformat.setDivisor(1)
            iformat.addColumn("offset", 4, Geom.NT_stdfloat, Geom.C_other)
            iformat.addColumn("rotation", 4, Geom.NT_stdfloat, Geom.C_other)  # Add rotation column
            iformat.addColumn("scale", 4, Geom.NT_stdfloat, Geom.C_other)  # Add scale column
            format.addArray(iformat)
            format = GeomVertexFormat.registerFormat(format)
            vdata.setFormat(format)

        sorted_positions = positionlist if zlevel is None else sorted(positionlist, key=lambda pos: pos[2], reverse=zlevel)
        if poswriter is None:
            poswriter = GeomVertexWriter(vdata.modifyArray(2), 0)
            rotwriter = GeomVertexWriter(vdata.modifyArray(2), 1)  # Writer for rotation
            scalewriter = GeomVertexWriter(vdata.modifyArray(2), 2)  # Writer for scale
            # print(new_location)

        self.instance_parent = NodePath('instances')
        for item in list(self.forplacementready.keys()):
            placeb = self.forplacementready.get(item)
            for posb, rotationb, scaleb, namesb in zip(placeb[0], placeb[1], placeb[2], placeb[3]):
                placeholder = self.instance_parent.attachNewNode(f'Instance_{namesb[0]}')
                group_xc = (posb[0] // 162) * 162
                group_yc = (posb[1] // 162) * 162
                group_keyc = (int(group_xc), int(group_yc))
                modelt, found_collider = self.search_loaded_files(namesb[0])
                # print(modelt)
                combined_keyc = f"{namesb[0]}{group_keyc}"
                if combined_keyc not in self.instance_groups:
                    self.instance_groups[combined_keyc] = [[], [], [], {}]
                self.instance_groups[combined_keyc][0].append([posb[0], posb[1], posb[2]])
                self.instance_groups[combined_keyc][1].append([rotationb[0], rotationb[1], rotationb[2], 1])
                self.instance_groups[combined_keyc][2].append([scaleb[0], scaleb[1], scaleb[2], 1])
                # poswriter.add_data3(posb[0], posb[1], posb[2])
                # rotwriter.add_data4(rotationb[0], rotationb[1], rotationb[2], 1)
                # scalewriter.add_data4(scaleb[0], scaleb[1], scaleb[2], 1)
                if found_collider is not None:
                    found_collider.hide()
                    scale_matrix = Mat4.scaleMat(scaleb[0], scaleb[1], scaleb[2])
                    rotation_matrixC = Mat4.rotateMat(-rotationb[0], LVector3f(0, 0, 1))
                    rotation_matrixB = Mat4.rotateMat(-rotationb[1], LVector3f(1, 0, 0))
                    rotation_matrixA = Mat4.rotateMat(-rotationb[2], LVector3f(0, 1, 0))
                    rotation_matrix = rotation_matrixC * rotation_matrixA * rotation_matrixB
                    translation_matrix = Mat4.translateMat(posb[0], posb[1], posb[2])
                    combined_matrix = scale_matrix * rotation_matrix * translation_matrix
                    placeholder.setMat(combined_matrix)
                    instance = found_collider.instanceTo(placeholder)
                    posi=instance.getPos(render)
                    hpri=instance.getHpr(render)
                    scalei=instance.getScale(render)
                    coordinate_location = f"{posb[0]}_{posb[1]}_{posb[2]}"
                    self.instance_groups[combined_keyc][3][str([posi[0], posi[1], posi[2]])+str([hpri[0], hpri[1], hpri[2]])+str([scalei[0],scalei[1],scalei[2]])]=[[posb[0], posb[1], posb[2]],[rotationb[0], rotationb[1], rotationb[2], 1],[scaleb[0], scaleb[1], scaleb[2], 1],instance]
                    instance.setTag('coordinate_location', f'{coordinate_location}_{indexB}_{namesb[0]}')
                    instance.setTag('placed','true')
                self.instance_parent.reparentTo(self.render)

                modelt.node().setBounds(OmniBoundingVolume())
                modelt.node().setFinal(True)
            del self.forplacementready[item]

        for i in range(value0, value1):
            skip_instanceb = False
            placeholder = self.instance_parent.attachNewNode(f'Instance_{i}')
            x, y, z = sorted_positions[i]
            pos = (x + new_location[0], y + new_location[1], z + new_location[2])
            placeholder.setPos(*pos)
            if directoyp == 'mountains' and regionmapsgot[x][y] == 13:
                skip_instanceb = False
            elif directoyp == 'desert' and regionmapsgot[x][y] == 255:
                skip_instanceb = False
            else:
                skip_instanceb = True

            if skip_instanceb:
                continue

            normal = self.calculate_normal(x, y, heightmapsgotfunc)
            group_x = (x // 162) * 162
            group_y = (y // 162) * 162
            group_key = (group_x + new_location[0], group_y + new_location[1])
            node_name = str(nodefunc.getName())
            combined_key = f"{node_name}{group_key}"
            if combined_key not in self.instance_groups:
                self.instance_groups[combined_key] = [[], [], [], {}]
            heading, pitch, _ = self.normal_to_euler(normal)
            rand = random.uniform(0, 180)
            if not roto:
                if np.degrees(pitch) >= 45:
                    h0, p0, r0 = rand, 0, 0
                else:
                    h0, p0, r0 = rand, np.degrees(pitch) / 2, np.degrees(heading) / 2
            else:
                h0, p0, r0 = rand, np.degrees(pitch), np.degrees(heading)
            scale = random.uniform(scales[0], scales[1])

            coordinate_location = LPoint3f(x + new_location[0], y + new_location[1], z + new_location[2])

            noten = self.forremoval.get(combined_key)
            if noten is not None:
                skip_instance = False
                for item in noten:
                    if item ==[[x + new_location[0], y + new_location[1], z + new_location[2]],[h0, p0, r0, 1],[scale, scale, scale, 1]]:
                        skip_instance = True
                        break
                if skip_instance:
                    continue
  
            self.instance_groups[combined_key][0].append([x + new_location[0], y + new_location[1], z + new_location[2]])
            self.instance_groups[combined_key][1].append([h0, p0, r0, 1])
            self.instance_groups[combined_key][2].append([scale, scale, scale, 1])
            # poswriter.add_data3(x + new_location[0], y + new_location[1], z + new_location[2])
            # rotwriter.add_data4(h0, p0, r0, 1)
            # scalewriter.add_data4(scale, scale, scale, 1)
            if nodecoll is not None:
                nodecoll.hide()
                scale_matrix = Mat4.scaleMat(scale, scale, scale)
                rotation_matrixC = Mat4.rotateMat(-h0, LVector3f(0, 0, 1))
                rotation_matrixB = Mat4.rotateMat(-p0, LVector3f(1, 0, 0))
                rotation_matrixA = Mat4.rotateMat(-r0, LVector3f(0, 1, 0))
                rotation_matrix = rotation_matrixC * rotation_matrixA * rotation_matrixB
                translation_matrix = Mat4.translateMat(x + new_location[0], y + new_location[1], z + new_location[2])
                combined_matrix = scale_matrix * rotation_matrix * translation_matrix  # Combine rotation and translation
                placeholder.setMat(combined_matrix)
                instance = nodecoll.instanceTo(placeholder)
                posi=instance.getPos(render)
                hpri=instance.getHpr(render)
                scalei=instance.getScale(render)
                coordinate_location = f"{x + new_location[0]}_{y + new_location[1]}_{z + new_location[2]}"
                self.instance_groups[combined_key][3][str([posi[0], posi[1], posi[2]])+str([hpri[0], hpri[1], hpri[2]])+str([scalei[0],scalei[1],scalei[2]])]=[[x + new_location[0], y + new_location[1], z + new_location[2]],[h0, p0, r0, 1],[scale, scale, scale, 1],instance]
                instance.setTag('coordinate_location', f'{coordinate_location}_{indexB}_{names}')

        if value0 == 0 and value1 ==0:
            
            # nodefunc.node().setBounds(OmniBoundingVolume())
            # nodefunc.node().setFinal(True)
            # nodefunc.setInstanceCount(0)
            pass
        else:
            self.instance_parent.reparentTo(self.render)
            total_instances += (value0 - value1)
            nodefunc.setInstanceCount(total_instances)
            nodefunc.node().setBounds(OmniBoundingVolume())
            nodefunc.node().setFinal(True)

        return total_instances, poswriter, rotwriter, scalewriter

    def setup_instance(self, heightmapsgotfunc, nodefunc, positionxyz, hprb, total_instances=0, poswriter=None, rotwriter=None, scalewriter=None, zrotonly=False, scaler=[],indexB=0,names=''):
        nodefunc, nodecoll = nodefunc
        gnode = nodefunc.find("**/+GeomNode").node()
        vdata = gnode.modifyGeom(0).modifyVertexData()
        format = GeomVertexFormat(gnode.getGeom(0).getVertexData().getFormat())
        if not format.hasColumn("offset"):
            iformat = GeomVertexArrayFormat()
            iformat.setDivisor(1)
            iformat.addColumn("offset", 4, Geom.NT_stdfloat, Geom.C_other)
            iformat.addColumn("rotation", 4, Geom.NT_stdfloat, Geom.C_other)
            iformat.addColumn("scale", 4, Geom.NT_stdfloat, Geom.C_other)
            format.addArray(iformat)
            format = GeomVertexFormat.registerFormat(format)
            vdata.setFormat(format)
        if poswriter is None:
            poswriter = GeomVertexWriter(vdata.modifyArray(2), 0)
        if rotwriter is None:
            rotwriter = GeomVertexWriter(vdata.modifyArray(2), 1)
        if scalewriter is None:
            scalewriter = GeomVertexWriter(vdata.modifyArray(2), 2)
        self.instance_parent = NodePath('instances')
        placeholder = self.instance_parent.attachNewNode(f'Instanceb_{0}')
        placeholder.setPos(positionxyz[0], positionxyz[1], positionxyz[2])
        group_x = (int(positionxyz[0]) // 162) * 162
        group_y = (int(positionxyz[1]) // 162) * 162
        group_key = (group_x, group_y)
        node_name = str(nodefunc.getName())
        combined_key = f"{node_name}{group_key}"
        if combined_key not in self.instance_groups:
            self.instance_groups[combined_key] = [[], [], [], {}]
        if combined_key not in self.forplacement:
            self.forplacement[combined_key] = [[], [], [], []]
        h0, p0, r0 = -hprb[0], -hprb[1], -hprb[2]
        scale=1
        coordinate_location = LPoint3f(positionxyz[0], positionxyz[1], positionxyz[2])
        self.instance_groups[combined_key][0].append([positionxyz[0], positionxyz[1], positionxyz[2]])
        self.instance_groups[combined_key][1].append([h0, p0, r0, 1])
        self.instance_groups[combined_key][2].append([scale, scale, scale, 1])
        self.forplacement[combined_key][0].append([positionxyz[0], positionxyz[1], positionxyz[2]])
        self.forplacement[combined_key][1].append([h0, p0, r0, 1])
        self.forplacement[combined_key][2].append([scale, scale, scale, 1])
        self.forplacement[combined_key][3].append([names])
        self.bufferdict3.append('1')
        self.forplacement=self.save_dict_as_json(self.world_name_p,self.forplacement, self.bufferdict3)
        if nodecoll is not None:
            nodecoll.hide()
            scale_matrix = Mat4.scaleMat(scale, scale, scale)
            rotation_matrixC = Mat4.rotateMat(-h0, LVector3f(0, 0, 1))
            rotation_matrixB = Mat4.rotateMat(-p0, LVector3f(1, 0, 0))
            rotation_matrixA = Mat4.rotateMat(-r0, LVector3f(0, 1, 0))
            rotation_matrix = rotation_matrixC * rotation_matrixA * rotation_matrixB
            translation_matrix = Mat4.translateMat(positionxyz[0], positionxyz[1], positionxyz[2])
            combined_matrix = scale_matrix * rotation_matrix * translation_matrix
            placeholder.setMat(combined_matrix)
            instance = nodecoll.instanceTo(placeholder)
            posi=instance.getPos(render)
            hpri=instance.getHpr(render)
            scalei=instance.getScale(render)
            self.instance_groups[combined_key][3][str([posi[0], posi[1], posi[2]])+str([hpri[0], hpri[1], hpri[2]])+str([scalei[0],scalei[1],scalei[2]])]=[[positionxyz[0],positionxyz[1],positionxyz[2]],[h0, p0, r0, 1],[scale, scale, scale, 1],instance]
            coordinate_location = f"{h0}_{p0}_{r0}"
            instance.setTag('coordinate_location', f'{coordinate_location}_{indexB}_{names}')
            instance.setTag('placed','true')
        self.instance_parent.reparentTo(self.render)
        nodefunc.node().setBounds(OmniBoundingVolume())
        nodefunc.node().setFinal(True)
        return total_instances, poswriter, rotwriter, scalewriter

    def generate_terrainCollide(self, height_map, size, position, scale_factor, texture_file,position2):

        format = GeomVertexFormat.getV3n3cpt2()
        vdata = GeomVertexData('vertices', format, Geom.UHStatic)
        vertex = GeomVertexWriter(vdata, 'vertex')
        normal = GeomVertexWriter(vdata, 'normal')
        color = GeomVertexWriter(vdata, 'color')
        texcoord = GeomVertexWriter(vdata, 'texcoord')
        vertices = []
        triangles = []
        rmap = (int(position.x  // 486 * 486),int(position.y // 486 * 486))
        
        # ... generate vertices, normals, colors, and texcoords based on height_map ...
        for y in range(size):
            for x in range(size):
  
                index_x = x * scale_factor + int(position.x) - rmap[0]
                index_y = y * scale_factor + int(position.y) - rmap[1]

                z = height_map[index_x, index_y]
                vertices.append((x * scale_factor, y * scale_factor, z))
                # Normalize the UV coordinates

                u_offset = (x * scale_factor + int(position.x) - rmap[0]) / 486
                v_offset = (y * scale_factor + int(position.y) - rmap[1]) / 486

                texcoord.addData2f(u_offset, v_offset)
                vertex.addData3f(x * scale_factor, y * scale_factor, z)

                # Calculate normal data
                normal_vec = self.calculate_normal(index_x, index_y, height_map)
                normal.addData3f(normal_vec)
                # Add color data
                color.addData4f(1, 1, 1, 1)  # Replace with actual color calculation

        geom = Geom(vdata)
        tris = GeomTriangles(Geom.UHStatic)
        
        # ... generate triangles ...
        for y in range(size - 1):
            for x in range(size - 1):
                i = x + y * size
                triangles.append((i, i+1, i+size))
                triangles.append((i+1, i+size+1, i+size))
                tris.addVertices(i, i+1, i+size)
                tris.addVertices(i+1, i+size+1, i+size)

        geom.addPrimitive(tris)
        texture = Texture()
        texture.load(texture_file)
        texture.setWrapU(Texture.WM_clamp)
        texture.setWrapV(Texture.WM_clamp)
        # Set texture filtering modes for pixel-perfect rendering
        texture.setMinfilter(Texture.FT_nearest)
        texture.setMagfilter(Texture.FT_nearest)
        node = GeomNode('gnode')
        node.addGeom(geom)
        entity = self.render.attachNewNode(node)

        # texture = texture_file
        ts = TextureStage('ts')
        ts.setMode(TextureStage.MModulate)
        entity.setTexture(ts, texture)
        entity.setPos(position)

        cnode = CollisionNode('cnode')
        for triangle in triangles:
            i1, i2, i3 = triangle
            v1 = Point3(vertices[i1][0], vertices[i1][1], vertices[i1][2])
            v2 = Point3(vertices[i2][0], vertices[i2][1], vertices[i2][2])
            v3 = Point3(vertices[i3][0], vertices[i3][1], vertices[i3][2])
            polygon = CollisionPolygon(v1, v2, v3)
            cnode.addSolid(polygon)
        entity.attachNewNode(cnode)

        entity.setTransparency(TransparencyAttrib.MAlpha)
        # Store the height map in the entity
        entity.setPythonTag("mappos",position2)
        entity.setPythonTag("height_map", height_map)
        entity.setPythonTag("size", size)
        entity.setPythonTag("hm", position)
        entity.setPythonTag("txtu", texture_file)

        entity.setShader(self.shader01)
        # entity.set_depth_offset(1)
        entity.set_bin('opaque', 1)
        return entity
    def generate_terrain(self,height_map, size, position, scale_factor, texture_file):
        format = GeomVertexFormat.getV3n3cpt2()
        vdata = GeomVertexData('vertices', format, Geom.UHStatic)
        vertex = GeomVertexWriter(vdata, 'vertex')
        normal = GeomVertexWriter(vdata, 'normal')
        color = GeomVertexWriter(vdata, 'color')
        texcoord = GeomVertexWriter(vdata, 'texcoord')
        vertices = []
        triangles = []

        rmap = (int(position.x  // 486) * 486,int(position.y // 486) * 486)

        # ... generate vertices, normals, colors, and texcoords based on height_map ...
        for y in range(size):
            for x in range(size):
                index_x = x * scale_factor + int(position.x) - rmap[0]
                index_y = y * scale_factor + int(position.y) - rmap[1]

                z = height_map[index_x, index_y]
                vertices.append((x * scale_factor, y * scale_factor, z))
                # Normalize the UV coordinates
                u_offset = ((x * scale_factor + int(position.x)) - rmap[0]) / 486
                v_offset = ((y * scale_factor + int(position.y)) - rmap[1]) / 486

                # uvs.append((u_offset, v_offset))
                vertex.addData3f(x * scale_factor, y * scale_factor, z)
                texcoord.addData2f(u_offset, v_offset)

                normal_vec = self.calculate_normal(index_x, index_y, height_map)
                normal.addData3f(normal_vec)
                color.addData4f(1, 1, 1, 1)  # Replace with actual color calculation
                # normal.addData3f(u_offset, v_offset, 1)

        geom = Geom(vdata)
        tris = GeomTriangles(Geom.UHStatic)

        # ... generate triangles ...
        for y in range(size - 1):
            for x in range(size - 1):
                i = x + y * size
                triangles.append((i, i+1, i+size))
                triangles.append((i+1, i+size+1, i+size))
                tris.addVertices(i, i+1, i+size)
                tris.addVertices(i+1, i+size+1, i+size)

        geom.addPrimitive(tris)

        node = GeomNode('gnode')
        node.addGeom(geom)

        # # Define the bounding volume for culling
        min_point = Point3(0, 0, min(height_map.flatten()))
        max_point = Point3(size * scale_factor, size * scale_factor, max(height_map.flatten()))
        bbox = BoundingBox(min_point, max_point)
        node.setBounds(bbox)
        node.setFinal(True)  # Ensures the bounding volume is not altered
        entity = self.render.attachNewNode(node)

        texture = Texture()
        
        texture.load(texture_file)
        texture.setWrapU(Texture.WM_clamp)
        texture.setWrapV(Texture.WM_clamp)

        # Set texture filtering modes for pixel-perfect rendering
        texture.setMinfilter(Texture.FT_nearest)
        texture.setMagfilter(Texture.FT_nearest)
        # texture = texture_file
        ts = TextureStage('ts')
        ts.setMode(TextureStage.MModulate)
        entity.setTexture(ts, texture)
        entity.setTransparency(TransparencyAttrib.MAlpha)
        entity.setPos(position)

        # Set the bin to render the terrain after the skybox but before other objects
        entity.setShader(self.shader01)
        entity.set_depth_offset(-4)
        entity.set_shader_input('enable_transparency',True)
        entity.set_bin('opaque', 1)
        return entity
    def value_range_generator(self, start, end, increment):#maybe useful?
        current_value0 = start
        current_value1 = start + increment
        
        while current_value0 < end:
            yield (current_value0, current_value1)
            current_value0 += increment
            current_value1 += increment
            
            if current_value1 > end:
                current_value1 = end
    def generate_terrain_if_needed(self, x_check, z_check):
        modified=[]
        # heightmapwas=False

        if x_check % 54 == 0 and x_check != self.holder[0]:
            self.holder[0] = x_check
            self.height_map_xz[0] = (x_check // 486) * 486
        if z_check % 54 == 0 and z_check != self.holder[1]:
            self.holder[1] = z_check
            self.height_map_xz[1] = (z_check // 486) * 486

        self.activatecont += 1
        if len(self.storework) > 0 and self.activatecont % 10 == 0:

            itm_index = self.activa % len(self.storework)
            self.activa += 1
            itm = self.storework[itm_index]
            for i, itemb in enumerate(self.loadedfiles):
                index = i * 6

                (self.total_instances,
                self.all_lists[index],
                self.all_lists[index+1],
                self.all_lists[index+2],
                    ) = self.setup_instancing(
                    itm[0],
                    itemb,
                    itm[2],
                    new_location=(itm[1][0], itm[1][1], 0),
                    zlevel=None,
                    total_instances=self.total_instances,
                    poswriter=self.all_lists[index],
                    rotwriter=self.all_lists[index+1],
                    scalewriter=self.all_lists[index+2],
                    indexB=index,names=self.modelnames[i],keycord=itm[3],directoyp=self.modeldirectory[i]
                    )
            del self.storework[itm_index]

        regionmapsgot = self.region_maps.get(tuple(self.height_map_xz))
        if regionmapsgot is None:
            region_map = np.zeros((487, 487))
            self.region_maps[tuple(self.height_map_xz)] = region_map

        heightmapsgot = self.height_maps.get(tuple(self.height_map_xz))
        if heightmapsgot is None:
            height_map = np.zeros((487, 487))
            self.height_maps[tuple(self.height_map_xz)] = height_map
            imagetx = PNMImage(512, 512)
            imagetx.read(Filename('data/level/Maintexture'+'.png'))
            imagetx.addAlpha()
            self.texture_maps[tuple(self.height_map_xz)] = imagetx
        # filelist=[]
        ent = self.checkdic.get(tuple(self.holder))
        # Define the range of offsets for checking neighboring tiles
        # offset_range = (-1458, -972, -486, 0, 486, 972, 1458)
        # Define a larger range of offsets for checking neighboring tiles
        # Define the offset step and range
        step = 486  # or any desired step value
        max_offset = 972  # maximum value for offset

        # Generate the offset range dynamically
        offset_rangeb = [i for i in range(-max_offset, max_offset + step, step)]

        # Loop through each offset to calculate the new heightmap keys
        for offset_x in offset_rangeb:
            for offset_z in offset_rangeb:
                # Calculate the new x and z keys based on the holder's position and the offset
                new_x_key = (self.holder[0] // 486) * 486 + offset_x
                new_z_key = (self.holder[1] // 486) * 486 + offset_z

                # Create a tuple for the heightmap key
                heightmap_key = (new_x_key, new_z_key)

                region_mapsgot = self.region_maps.get(heightmap_key)
                if region_mapsgot is None:
                    region_map = np.zeros((487, 487))
                    self.region_maps[heightmap_key] = region_map
                # Check if the heightmap key exists in the height_maps dictionary
                heightmapsgot = self.height_maps.get(heightmap_key)
                if heightmapsgot is None:

                    modified.append(heightmap_key)
                    height_map = np.zeros((487, 487))
                    self.height_maps[heightmap_key] = height_map
                    imagetx = PNMImage(512, 512)
                    imagetx.read(Filename('data/level/Maintexture'+'.png'))
                    imagetx.addAlpha()
                    self.texture_maps[heightmap_key] = imagetx

                if (new_x_key,new_z_key) in modified:
                    randomized=[]
                    parent_folder = 'data/level/'
                    savedworld=self.savesedit.get(str((new_x_key,new_z_key)))

                    if savedworld is not None:
                        for i in savedworld:
                            file_names = [i[1]+'/'+i[2]]
                            # print(parent_folder+i[0], file_names,'filenames')
                            strength=i[6]
                            du=i[7]
                            inverter=i[8]
                            if i[0] == 'terrains':
                                file = self.load_specific_files(parent_folder, file_names, self.filelist)
                                self.brush_height2, self.bigtest, filepath0, filepath1, self.regionfile01 = file[0]

                                original_bigtest = copy.deepcopy(self.bigtest)
                                original_brush_height2 = copy.deepcopy(self.brush_height2)
                                original_regionfile01= copy.deepcopy(self.regionfile01)
                                num_turns = i[3]
                                if num_turns == 0:
                                    flip_x, flip_y, transpose = False, False, False
                                elif num_turns == 1:
                                    flip_x, flip_y, transpose = False, True, True
                                elif num_turns == 2:
                                    flip_x, flip_y, transpose = True, True, False
                                else:
                                    flip_x, flip_y, transpose = True, False, True
                                self.bigtest.flip(flip_x, flip_y, transpose)
                                self.brush_height2 = np.rot90(self.brush_height2, num_turns)
                                self.regionfile01 = np.rot90(self.regionfile01, num_turns)
                                coordinate_x = i[4]
                                coordinate_y = i[5]
                                self.process_heightmaps(new_x_key, new_z_key, coordinate_x, coordinate_y, scale_factor,self.bigtest,self.brush_height2,strength,self.regionfile01,True,du,inverter)
                                # self.process_regionmaps(new_x_key, new_z_key, coordinate_x, coordinate_y,regionfile,False)

                                self.bigtest = copy.deepcopy(original_bigtest)
                                self.brush_height2 = copy.deepcopy(original_brush_height2)
                                self.regionfile01 = copy.deepcopy(original_regionfile01)
                    else:
                        existing_coords = []
                        random_boolean = random.choice([True, False])
                        for i in range(10):
                            distchfile=self.get_terrain_file(new_x_key,new_z_key, parent_folder)
                            # print([distchfile[0]+'/'+distchfile[1]],'rand')
                            fileb = self.load_specific_files(parent_folder+'terrains/', [distchfile[0]+'/'+distchfile[1]], self.filelist)
                            # print(fileb)
                            # if distchfile[0] == "desert":

                            #     print(fileb[0],distchfile[0])
                            strength = 10
                            du = False
                            self.brush_height2, self.bigtest, filepath0, filepath1, self.regionfile01 = fileb[0]
                            original_bigtest = copy.deepcopy(self.bigtest)
                            original_brush_height2 = copy.deepcopy(self.brush_height2)
                            original_regionfile01= copy.deepcopy(self.regionfile01)
                            num_turns = random.randint(0, 3)
                            if num_turns == 0:
                                flip_x, flip_y, transpose = False, False, False
                            elif num_turns == 1:
                                flip_x, flip_y, transpose = False, True, True
                            elif num_turns == 2:
                                flip_x, flip_y, transpose = True, True, False
                            else:
                                flip_x, flip_y, transpose = True, False, True
                            self.bigtest.flip(flip_x, flip_y, transpose)

                            self.brush_height2 = np.rot90(self.brush_height2, num_turns)
                            self.regionfile01 = np.rot90(self.regionfile01, num_turns)
                            offset_range0 = 243
                            min_distance = 100
                            while True:
                                coordinate_x = random.randint(new_x_key - offset_range0, new_x_key + offset_range0)
                                coordinate_y = random.randint(new_z_key - offset_range0, new_z_key + offset_range0)
                                if self.is_far_enough(coordinate_x, coordinate_y, existing_coords, min_distance):
                                    existing_coords.append((coordinate_x, coordinate_y))
                                    break
                            randomized.append(['terrains', filepath0, filepath1, num_turns, coordinate_x, coordinate_y, strength, du, random_boolean])
                            self.process_heightmaps(new_x_key, new_z_key, coordinate_x, coordinate_y, scale_factor,self.bigtest,self.brush_height2,strength,self.regionfile01,True,du,random_boolean)
                            # self.process_regionmaps(new_x_key, new_z_key, coordinate_x, coordinate_y,regionfile,False)
                            self.bigtest = copy.deepcopy(original_bigtest)
                            self.brush_height2 = copy.deepcopy(original_brush_height2)
                            self.regionfile01 = copy.deepcopy(original_regionfile01)

                        self.savesedit[str((new_x_key, new_z_key))] = randomized
                        self.bufferdict.append('1')

            self.savesedit=self.save_dict_as_json(self.world_name_n,self.savesedit, self.bufferdict)
            for bz in (-972,-486, 0, 486,972):
                for bx in (-972,-486, 0, 486,972):
                    new_x = (self.holder[0] // 161) * 162 + bx
                    new_z = (self.holder[1] // 161) * 162 + bz
                    new_x_key = (new_x) // 486 * 486
                    new_z_key = (new_z) // 486 * 486
                    if (new_x_key,new_z_key) in modified:
                        parent_folder = 'data/level/'
                        savedworld=self.savesedit.get(str((new_x_key,new_z_key)))
                        if savedworld is not None:
                            for i in savedworld:
                                file_names = [i[0]+'/'+i[1]+'/'+i[2]]
                                strength=i[6]
                                du=i[7]
                                inverter=i[8]
                                if i[0] == 'handbrushes':
                                    file = self.load_specific_files(parent_folder, file_names, self.filelist)
                                    # print(parent_folder,[i[0]+'/'+i[1]+'/'+i[2]],strength,du,inverter)
                                    self.brush_height2, self.bigtest, filepath0, filepath1, self.regionfile01  = file[0]
                                    original_bigtest = copy.deepcopy(self.bigtest)
                                    original_brush_height2 = copy.deepcopy(self.brush_height2)
                                    # original_regionfile = copy.deepcopy(regionfile)
                                    num_turns = i[3]
                                    if num_turns == 0:
                                        flip_x, flip_y, transpose = False, False, False
                                    elif num_turns == 1:
                                        flip_x, flip_y, transpose = False, True, True
                                    elif num_turns == 2:
                                        flip_x, flip_y, transpose = True, True, False
                                    else:
                                        flip_x, flip_y, transpose = True, False, True
                                    self.bigtest.flip(flip_x, flip_y, transpose)
                                    self.brush_height2 = np.rot90(self.brush_height2, num_turns)
                                    coordinate_x = i[4]
                                    coordinate_y = i[5]
                                    self.process_heightmaps(new_x_key, new_z_key, coordinate_x, coordinate_y, scale_factor,self.bigtest,self.brush_height2,strength,self.regionfile01,False,du,inverter)
                                    self.bigtest = copy.deepcopy(original_bigtest)
                                    self.brush_height2 = copy.deepcopy(original_brush_height2)

        if ent is None:
            heightmapsgot = self.height_maps.get(tuple(self.height_map_xz))

            txture = self.texture_maps.get(tuple(self.height_map_xz))
            terrain_s3 = self.generate_terrainCollide(heightmapsgot, 19, Vec3(self.holder[0],self.holder[1], 0), 3, txture,tuple(self.height_map_xz))#should be above generate_terrain
            self.checkdic[tuple(self.holder)] = terrain_s3

            for offset_x in (-324,-162, 0, 162,324):
                for offset_z in (-324,-162, 0, 162,324):
                    new_xc = (self.holder[0] // 161) * 162 + offset_x
                    new_zc = (self.holder[1] // 161) * 162 + offset_z
                    new_x_keyc = (new_xc) // 486 * 486
                    new_z_keyc = (new_zc) // 486 * 486
                    heightmap_keyc = (new_x_keyc , new_z_keyc)
                    heightmapsgot = self.height_maps.get(heightmap_keyc)
                    objentities=self.objmap.get(str(heightmap_keyc))
                    regionmapsgot = self.region_maps.get(tuple(heightmap_keyc))

                    if objentities is None:
                        positionlist=self.position_gen(5000,12,486,486,heightmapsgot)


                        self.objmap[str(heightmap_keyc)]=positionlist
                        self.storework.append([heightmapsgot,heightmap_keyc,positionlist,heightmap_keyc])

                    txture = self.texture_maps.get(heightmap_keyc)
                    bigent = self.dictn.get((new_xc, new_zc))
                    if bigent is None:
                        terrain_s3 = self.generate_terrain(heightmapsgot, 28, Vec3(new_xc, new_zc, -0.1), 6, txture)
                        self.dictn[(new_xc, new_zc)] = terrain_s3
                        
    def is_far_enough(self, coordinate_x, coordinate_y, existing_coords, min_distance):
        for (x, y) in existing_coords:
            if math.hypot(coordinate_x - x, coordinate_y - y) < min_distance:
                return False
        return True
    def remove_far_meshes0(self,radius):
        #self.dictn.keys()
        for key in list(self.checkdic.keys()):  # We use list to avoid 'dictionary changed size during iteration' error
            mesh = self.checkdic[key]

            # Assuming self.holder is a list or tuple of coordinates
            point = Vec3(self.holder[0], self.holder[1], 0)
            # Get the position of the mesh
            mesh_pos = mesh.getPos()
            # Calculate the distance between the point and the mesh
            distance = (point - mesh_pos).length()

            if distance > radius:
                mesh.removeNode()
                del self.checkdic[key]  # remove the reference from the dictionary
    def remove_far_meshes1(self,radius):
        for key in list(self.dictn.keys()):  # We use list to avoid 'dictionary changed size during iteration' error
            mesh = self.dictn[key]

            # Assuming self.holder is a list or tuple of coordinates
            point = Vec3(self.holder[0], self.holder[1], 0)

            # Get the position of the mesh
            mesh_pos = mesh.getPos()

            # Calculate the distance between the point and the mesh
            distance = (point - mesh_pos).length()

            # If the distance is greater than the radius, remove the mesh and delete the key
            if distance > radius:
                mesh.removeNode()
                del self.dictn[key]  # remove the reference from the dictionary
    def save_dict_as_json(self, file_path, mera, buffer):
        # Check if the file exists before attempting to read it
        if len(mera) <= 0:

            if os.path.exists(file_path):
                with open(file_path, 'r') as openfile:
                    json_object = json.load(openfile)
                    mera = json_object
            else:
                mera = {}
        if len(buffer) >= 2:
            json_object = json.dumps(mera)
            with open(file_path, "w") as outfile:
                outfile.write(json_object)
            buffer.clear()
        return mera
    def process_heightmaps(self, x, z, xp, zp, scale_factor,imgbrush,brush,divide,regionfile,regtog=False,terrainup=True, inverted=False):
        offset = 54
        negetive=len(brush[0])//2
        modified_heightmaps = set()
        # Neighbors
        neighbors = [
            (int((x + offset) // 486) * 486, int(z // 486) * 486),  # up
            (int((x - offset) // 486) * 486, int(z // 486) * 486),  # down
            (int(x // 486) * 486, int((z + offset) // 486) * 486),  # right
            (int(x // 486) * 486, int((z - offset) // 486) * 486),  # left
            (int((x + offset) // 486) * 486, int((z + offset) // 486) * 486),  # top right corner
            (int((x - offset) // 486) * 486, int((z + offset) // 486) * 486),  # top left corner
            (int((x + offset) // 486) * 486, int((z - offset) // 486) * 486),  # bottom right corner
            (int((x - offset) // 486) * 486, int((z - offset) // 486) * 486),   # bottom left corner
            (int(x  // 486) * 486,int(z // 486) * 486)
        ]
        for hmap in neighbors:
            if hmap not in modified_heightmaps:
                zk=int(zp) - hmap[1]
                zk-=negetive
                xk=int(xp) - hmap[0]
                xk-=negetive
                heightmapsgot = self.height_maps.get(hmap)
                regionmapsgot = self.region_maps.get(hmap)
                txturee = self.texture_maps.get(tuple(hmap))
                if xk >= -negetive-negetive and xk <= 486 and zk >= -negetive-negetive and zk <= 486:
                    if heightmapsgot is None:
                        height_map = np.zeros((487, 487))
                        self.height_maps[tuple(hmap)] = height_map
                        imagetx = PNMImage(512, 512)
                        imagetx.read(Filename('data/level/Maintexture'+'.png'))
                        imagetx.addAlpha()
                        self.texture_maps[tuple(hmap)] = imagetx
                    if heightmapsgot is not None:
                        heightmapsgot = self.painter(heightmapsgot, brush / divide, zk, xk, inverted)
                        self.height_maps[hmap] = heightmapsgot
                        modified_heightmaps.add(hmap)
                    if regtog == True:
                        if regionmapsgot is not None:
                            regionmapsgot = self.painterreplace(regionmapsgot, regionfile, zk, xk, False)
                            self.region_maps[hmap] = regionmapsgot
                    if txturee is not None:
                        txturee.blendSubImage(imgbrush, int(xk*scale_factor),int(512-zk*scale_factor)-negetive-negetive)
                        self.textureC.load(txturee)
                    if terrainup == True:
                        for offset_x in (0, 162, 324, 486):
                            for offset_z in (0, 162, 324, 486):
                                new_x = (tuple(hmap)[0] // 161) * 162 + offset_x
                                new_z = (tuple(hmap)[1] // 161) * 162 + offset_z
                                bigent=self.dictn.get((new_x,new_z))
                                if bigent is not None:
                                    heightmapsgot = self.height_maps.get(tuple(hmap))
                                    txture = self.texture_maps.get(tuple(hmap))
                                    bigent.removeNode()
                                    terrain_s3=self.generate_terrainCollide(heightmapsgot, 28, Vec3((new_x,new_z), -0.5), 6, txture, heightmapsgot)
                                    coord_tuplevi = ((new_x,new_z))
                                    self.dictn[coord_tuplevi] = terrain_s3



    def get_full_path(self, node_path):
        full_path = ''
        while node_path != render and not node_path.isEmpty():
            full_path = '/' + node_path.getName() + full_path
            node_path = node_path.getParent()
        return 'render' + full_path

    def check_for_instance_name(self, node_path, desired_name):
        # Get the full path of the node by traversing the parents
        full_path = self.get_full_path(node_path)
        
        # Check if the full path contains the desired name
        if desired_name in full_path:
            return True
        else:
            return False



    def raycastlight(self, max_distance=7, normal_offset=0.1):
        if base.mouseWatcherNode.hasMouse():
            camera_position = base.camera.getPos(render)
            # Convert the 2D screen space coordinates to 3D ray
            mpos = base.mouseWatcherNode.getMouse()
            pFrom = Point3()
            pTo = Point3()
            base.camLens.extrude(mpos, pFrom, pTo)

            # Convert the points from camera space to world space
            world_pFrom = render.getRelativePoint(base.cam, pFrom)
            world_pTo = render.getRelativePoint(base.cam, pTo)

            # Calculate the direction vector from the player's position to the mouse position
            direction = world_pTo - world_pFrom
            direction.normalize()  # Normalize the direction vector

            # Scale the direction vector by the maximum distance
            direction *= max_distance

            self.rayC.setOrigin(camera_position)
            self.rayC.setDirection(direction)
            # Perform the duplicate raycast
            self.traverser.traverse(render)

            # Calculate the end position of the raycast
            end_pos = camera_position + direction

            if self.queuec.getNumEntries() > 0:
                self.queuec.sortEntries()
                hit = self.queuec.getEntry(0)
                hitPos = hit.getSurfacePoint(render)
                # Check if the hit position is within the max distance
                distance_to_hit = (hitPos - camera_position).length()
                if distance_to_hit <= max_distance:
                    # Move the hit position away from the normals of the hit object
                    normal = hit.getSurfaceNormal(render)
                    hitPos += normal * normal_offset
                    self.point_path.setPos(hitPos)
                else:
                    self.point_path.setPos(end_pos)
            else:
                # Set the end position of the raycast as the point position if no hit is detected
                self.point_path.setPos(end_pos)



    def raycastangle(self):
        if base.mouseWatcherNode.hasMouse():
            # Get the player's position
            camera_position = base.camera.getPos(render)
            # Get the mouse position in 2D screen space
            mpos = base.mouseWatcherNode.getMouse()

            # Convert the 2D screen space coordinates to 3D ray
            pFrom = Point3()
            pTo = Point3()
            base.camLens.extrude(mpos, pFrom, pTo)

            # Convert the points from camera space to world space
            world_pFrom = render.getRelativePoint(base.cam, pFrom)
            world_pTo = render.getRelativePoint(base.cam, pTo)

            # Calculate the direction vector from the player's position to the mouse position
            direction = world_pTo - world_pFrom

            # Set the origin and direction of the primary ray
            self.ray.setOrigin(camera_position)
            self.ray.setDirection(direction)

            # Perform the primary raycast
            self.traverser.traverse(render)
            if self.queue.getNumEntries() > 0:
                self.queue.sortEntries()
                hit = self.queue.getEntry(0)

                hitNodePath = hit.getIntoNodePath()
                hitPos = hit.getSurfacePoint(render)
                # self.moveThroughBoxesb.setPos(hitPos[0], hitPos[1], hitPos[2])
                entity = hitNodePath.getParent()

                xp, zp, yp = hitPos
                xe, ze, ye = entity.getPos()
                xer, zer, yer = entity.getPos(render)
                normal = hit.getSurfaceNormal(render)
                # Assuming self.moveThroughBoxesb is already defined

                self.ray01.setOrigin(camera_position)
                self.ray01.setDirection(direction)
                # Perform the duplicate raycast
                self.traverser.traverse(render)
                ghost_entry = None
                if self.queueb.getNumEntries() > 0:
                    self.queueb.sortEntries()

                    for entry in self.queueb.entries:
                        if entry.getIntoNodePath().getName() == "Ghosts":
                            ghost_entry = entry
                            break  # Exit the loop as soon as we find the ghost entry


                x,z = int(xe),int(ze)

                if self.keynormalstog == True:
    
                    # heading, pitch, roll = self.normal_to_euler(normal)
                    heading = base.camera.getHpr(render)[0]

                    # Normalize the heading to a range of 0 to 360 degrees
                    normalized_heading = heading % 360

                    # Determine the facing direction based on the heading value and set the new position to the right of the facing direction
                    if 45 <= normalized_heading < 135:
                        direction = "East"
                        pitch0=0
                        roll0=-45
                        # self.rotationsmem[2]-=45
                        # self.pointer.setHpr(rot[0],rot[1],rot[2]-45)
                        # self.pointer.setPos(pos[0], pos[1] - 1 - self.incre, pos[2])  # Move to the right of East (South)
                    elif 135 <= normalized_heading < 225:
                        direction = "South"
                        pitch0=45
                        roll0=0
                        # self.rotationsmem[1]+=45
                        # self.pointer.setHpr(rot[0],rot[1]+45,rot[2])
                        # self.pointer.setPos(pos[0] + 1 + self.incre, pos[1], pos[2])  # Move to the right of South (West)
                    elif 225 <= normalized_heading < 315:
                        direction = "West"
                        pitch0=0
                        roll0=45
                        # self.rotationsmem[2]+=45
                        # self.pointer.setHpr(rot[0],rot[1],rot[2]+45)
                        # self.pointer.setPos(pos[0], pos[1] + 1 + self.incre, pos[2])  # Move to the right of West (North)
                    else:
                        direction = "North"
                        pitch0=-45
                        roll0=0
                        # self.rotationsmem[1]-=45
                        # self.pointer.setHpr(rot[0],rot[1]-45,rot[2])
                    h0, p0, r0 = np.degrees(0), np.degrees(pitch0), np.degrees(roll0)
                    r0+=self.rotationsmem[2]
                    p0+=self.rotationsmem[1]
                    h0+=self.rotationsmem[0]
                    scale_matrix = Mat4.scaleMat(0.1, 0.1, 0.1)
                    rotation_matrixC = Mat4.rotateMat(-h0, LVector3f(0, 0, 1))
                    rotation_matrixB = Mat4.rotateMat(-p0, LVector3f(1, 0, 0))
                    rotation_matrixA = Mat4.rotateMat(-r0, LVector3f(0, 1, 0))
                    rotation_matrix = rotation_matrixC * rotation_matrixA * rotation_matrixB
                    translation_matrix = Mat4.translateMat(xp, zp, yp+self.zmem)
                    combined_matrix = scale_matrix * rotation_matrix * translation_matrix  # Combine rotation and translation
                    self.pointer.setMat(combined_matrix)
                    # self.pointer.setHpr(0,np.degrees(pitch),np.degrees(heading))
                    
                    if self.keypointhidetog == True:
                        self.pointer.show()
                else:
                    if entity.hasTag('placed'):
                        
                        bounds = entity.getTightBounds()

                        # Calculate the dimensions
                        min_point, max_point = bounds
                        width = max_point.getX() - min_point.getX()
                        height = max_point.getZ() - min_point.getZ()
                        depth = max_point.getY() - min_point.getY()

                        boundsdisplay = self.todisplay.getTightBounds()

                        # Calculate the dimensions
                        min_pointdisplay, max_pointdisplay = boundsdisplay
                        widthdisplay = max_pointdisplay.getX() - min_pointdisplay.getX()
                        heightdisplay = max_pointdisplay.getZ() - min_pointdisplay.getZ()
                        depthdisplay = max_pointdisplay.getY() - min_pointdisplay.getY()

                        rotation = entity.getHpr(render)
                        xyz=[0,0,0]
                        position = entity.getPos(render)
                        if entity != self.heldentity:
                            self.heldentity= entity
                            
                            # rotation = entity.getHpr(render)
                            # To clear all instances on the parent node
                            self.new_parent_node.getChildren().detach()
                            # List of positions to loop through

                            if height == 4:
                                if int(heightdisplay) == 0:
                                    positions = [
                                        [2, 0, 3],
                                        [-2, 0, 3],
                                        [0, 2, 3],
                                        [0, -2, 3],
                                        [0, 0, 4],
                                        [0, 0, 0],

                                        [2, 0, 2],
                                        [-2, 0, 2],
                                        [0, 2, 2],
                                        [0, -2, 2],

                                        [2, 0, 1],
                                        [-2, 0, 1],
                                        [0, 2, 1],
                                        [0, -2, 1],

                                        [2, 0, 0],
                                        [-2, 0, 0],
                                        [0, 2, 0],
                                        [0, -2, 0],
                                    ]
                                else:
                                    positions = [
                                        [2, 0, 0],
                                        [-2, 0, 0],
                                        [0, 2, 0],
                                        [0, -2, 0],
                                        [0, 0, 4],
                                        [0, 0, 0],

                                        [2, 0, 2],
                                        [-2, 0, 2],
                                        [0, 2, 2],
                                        [0, -2, 2],
                                    ]
                            elif int(height) == 0:
                                positions = [
                                    [2, 0, 0],
                                    [-2, 0, 0],
                                    [0, 2, 0],
                                    [0, -2, 0],
                                    [0, 0, int(height)+1],
                                    [0, 0, 0]
                                ]

                            # else:
                            #     positions = [
                            #         [2, 0, 0],
                            #         [-2, 0, 0],
                            #         [0, 2, 0],
                            #         [0, -2, 0],
                            #         [0, 0, int(height)],
                            #         [0, 0, 0]
                            #     ]

                            for xyz in positions:

                                # Define the movement vector in the local Z direction
                                local_movement = Vec3(xyz[0], xyz[1], xyz[2])  # Adjust the magnitude as needed

                                # Transform the movement vector to world space using the entity's transformation matrix
                                movement_vector = entity.getQuat(render).xform(local_movement)

                                # Update the position with the transformed movement vector
                                new_position = position + movement_vector
                                placeholder = self.new_parent_node.attachNewNode(f'Instanceb_{movement_vector}')
                                placeholder.setPos(new_position)
                                placeholder.setHpr(rotation)
                                if int(height) == 0:
                                    print(xyz[0], xyz[1], xyz[2])
                                    if xyz[1] == 2:
                                        instance = self.moveThroughBoxesh1.instanceTo(placeholder)
                                    elif xyz[1] == -2:
                                        instance = self.moveThroughBoxesh2.instanceTo(placeholder)
                                    elif xyz[0] == 2:
                                        instance = self.moveThroughBoxesh3.instanceTo(placeholder)
                                    elif xyz[0] == -2:
                                        instance = self.moveThroughBoxesh4.instanceTo(placeholder)
                                    else:
                                        instance = self.moveThroughBoxesh.instanceTo(placeholder)
                                else:
                                    if int(heightdisplay) == 0:
                                        print(self.rotationsmem)
             
                                        instance = self.moveThroughBoxesh.instanceTo(placeholder)
                                    else:
                                        instance = self.moveThroughBoxesb.instanceTo(placeholder)
                                

                        if ghost_entry:
                            ghost_pos = ghost_entry.getIntoNodePath().getPos(render)
                            # Example usage with the collision entry

                            desired_name = "Instanceb_LVecBase3f(0, 0, 0)"
                            if self.check_for_instance_name(ghost_entry.getIntoNodePath(), desired_name):

                                if int(heightdisplay) == 0:
                                    local_movement = Vec3(0, 0, -1)
                                else:
                                    local_movement = Vec3(0, 0, -int(height))  # Adjust the magnitude as needed

                                # Transform the movement vector to world space using the entity's transformation matrix
                                movement_vector = entity.getQuat(render).xform(local_movement)

                                # Update the position with the transformed movement vector
                                new_position = position + movement_vector
                                self.pointer.setPos(new_position[0],new_position[1],new_position[2]+self.zmem)
                            else:
                                self.pointer.setPos(ghost_pos[0],ghost_pos[1],ghost_pos[2]+self.zmem)

                        if self.keypointhidetog == True:
                            self.pointer.show()

                    else:
                        # self.zmem = 0
                        self.new_parent_node.getChildren().detach()
                        self.heldentity=''
                        self.pointer.setPos(int(xp), int(zp), int(yp))
                        if self.keypointhidetog == True:
                            self.pointer.show()


    # Define a function to perform the raycast
    def performRaycastRight(self):
        if base.mouseWatcherNode.hasMouse():
            mpos = base.mouseWatcherNode.getMouse()

            # Convert the 2D screen space coordinates to 3D ray
            pFrom = Point3()
            pTo = Point3()
            base.camLens.extrude(mpos, pFrom, pTo)

            # Convert the points from camera space to world space
            world_pFrom = render.getRelativePoint(base.cam, pFrom)
            world_pTo = render.getRelativePoint(base.cam, pTo)

            # Calculate the direction vector from the player's position to the mouse position
            direction = world_pTo - world_pFrom
            # Retrieve the camera's position
            camera_position = base.camera.getPos(render)


            # Set the origin and direction of the ray
            self.ray.setOrigin(camera_position)
            self.ray.setDirection(direction)

            # Perform the raycast
            self.traverser.traverse(render)
            if self.queue.getNumEntries() > 0:
                self.queue.sortEntries()
                hit = self.queue.getEntry(0)

                hitNodePath = hit.getIntoNodePath()
                hitPos = hit.getSurfacePoint(render)
                normal = hit.getSurfaceNormal(self.render)

                entity = hitNodePath.getParent()
                xp,zp,yp=hitPos
                xe,ze,ye= entity.getPos()
                xer,zer,yer= entity.getPos(render)

                self.ray01.setOrigin(camera_position)
                self.ray01.setDirection(direction)
                # Perform the duplicate raycast
                self.traverser.traverse(render)
                ghost_entry = None
                if self.queueb.getNumEntries() > 0:
                    self.queueb.sortEntries()

                    for entry in self.queueb.entries:
                        if entry.getIntoNodePath().getName() == "Ghosts":
                            ghost_entry = entry
                            break  # Exit the loop as soon as we find the ghost entry

                x,z = int(xe),int(ze)
                hmap0=(int(x  // 486) * 486,int(z // 486) * 486)

                if self.key2tog:
                    if self.keynormalstog == True:
                        # heading, pitch, roll = self.normal_to_euler(normal)
                        heading = base.camera.getHpr(render)[0]

                        # Normalize the heading to a range of 0 to 360 degrees
                        normalized_heading = heading % 360

                        # Determine the facing direction based on the heading value and set the new position to the right of the facing direction
                        if 45 <= normalized_heading < 135:
                            direction = "East"
                            pitch0=0
                            roll0=-45
                            # self.rotationsmem[2]-=45
                            # self.pointer.setHpr(rot[0],rot[1],rot[2]-45)
                            # self.pointer.setPos(pos[0], pos[1] - 1 - self.incre, pos[2])  # Move to the right of East (South)
                        elif 135 <= normalized_heading < 225:
                            direction = "South"
                            pitch0=45
                            roll0=0
                            # self.rotationsmem[1]+=45
                            # self.pointer.setHpr(rot[0],rot[1]+45,rot[2])
                            # self.pointer.setPos(pos[0] + 1 + self.incre, pos[1], pos[2])  # Move to the right of South (West)
                        elif 225 <= normalized_heading < 315:
                            direction = "West"
                            pitch0=0
                            roll0=45
                            # self.rotationsmem[2]+=45
                            # self.pointer.setHpr(rot[0],rot[1],rot[2]+45)
                            # self.pointer.setPos(pos[0], pos[1] + 1 + self.incre, pos[2])  # Move to the right of West (North)
                        else:
                            direction = "North"
                            pitch0=-45
                            roll0=0
                            # self.rotationsmem[1]-=45
                            # self.pointer.setHpr(rot[0],rot[1]-45,rot[2])

                        h0, p0, r0 = np.degrees(0), np.degrees(pitch0), np.degrees(roll0)
                        r0+=self.rotationsmem[2]
                        p0+=self.rotationsmem[1]
                        h0+=self.rotationsmem[0]
                        self.phpr=[-h0, -p0, -r0]
                        scale_matrix = Mat4.scaleMat(0.1, 0.1, 0.1)
                        rotation_matrixC = Mat4.rotateMat(-h0, LVector3f(0, 0, 1))
                        rotation_matrixB = Mat4.rotateMat(-p0, LVector3f(1, 0, 0))
                        rotation_matrixA = Mat4.rotateMat(-r0, LVector3f(0, 1, 0))
                        rotation_matrix = rotation_matrixC * rotation_matrixA * rotation_matrixB
                        translation_matrix = Mat4.translateMat(xp, zp, yp+self.zmem)
                        combined_matrix = scale_matrix * rotation_matrix * translation_matrix  # Combine rotation and translation
                        self.pointer.setMat(combined_matrix)
                        if self.keypointhidetog == True:
                            self.pointer.show()
                        self.placefunc()
                    else:
                        if entity.hasTag('placed'):
                            # Calculate movement vector based on the normal
                            bounds = entity.getTightBounds()

                            # Calculate the dimensions
                            min_point, max_point = bounds
                            width = max_point.getX() - min_point.getX()
                            height = max_point.getZ() - min_point.getZ()
                            depth = max_point.getY() - min_point.getY()

                            boundsdisplay = self.todisplay.getTightBounds()

                            # Calculate the dimensions
                            min_pointdisplay, max_pointdisplay = boundsdisplay
                            widthdisplay = max_pointdisplay.getX() - min_pointdisplay.getX()
                            heightdisplay = max_pointdisplay.getZ() - min_pointdisplay.getZ()
                            depthdisplay = max_pointdisplay.getY() - min_pointdisplay.getY()


                            # print("Position:", position)
                            # print("Width:", width)
                            # print("Height:", height)
                            # print("Depth:", depth)

    
                            rotation = entity.getHpr(render)
                            xyz=[0,0,0]
                            position = entity.getPos(render)
                            if entity != self.heldentity:
                                self.heldentity= entity
                                
                                rotation = entity.getHpr(render)
                                # To clear all instances on the parent node
                                self.new_parent_node.getChildren().detach()
                                # List of positions to loop through

                                if height == 4:
                                    if int(heightdisplay) == 0:
                                        positions = [
                                            [2, 0, 3],
                                            [-2, 0, 3],
                                            [0, 2, 3],
                                            [0, -2, 3],
                                            [0, 0, 4],
                                            [0, 0, 0],

                                            [2, 0, 2],
                                            [-2, 0, 2],
                                            [0, 2, 2],
                                            [0, -2, 2],

                                            [2, 0, 1],
                                            [-2, 0, 1],
                                            [0, 2, 1],
                                            [0, -2, 1],

                                            [2, 0, 0],
                                            [-2, 0, 0],
                                            [0, 2, 0],
                                            [0, -2, 0],
                                        ]
                                    else:
                                        positions = [
                                            [2, 0, 0],
                                            [-2, 0, 0],
                                            [0, 2, 0],
                                            [0, -2, 0],
                                            [0, 0, 4],
                                            [0, 0, 0],

                                            [2, 0, 2],
                                            [-2, 0, 2],
                                            [0, 2, 2],
                                            [0, -2, 2],
                                        ]
                                elif int(height) == 0:
                                    positions = [
                                        [2, 0, 0],
                                        [-2, 0, 0],
                                        [0, 2, 0],
                                        [0, -2, 0],
                                        [0, 0, int(height)+1],
                                        [0, 0, 0]
                                    ]

                                # else:
                                #     positions = [
                                #         [2, 0, 0],
                                #         [-2, 0, 0],
                                #         [0, 2, 0],
                                #         [0, -2, 0],
                                #         [0, 0, int(height)],
                                #         [0, 0, 0]
                                #     ]

                                for xyz in positions:

                                    # Define the movement vector in the local Z direction
                                    local_movement = Vec3(xyz[0], xyz[1], xyz[2])  # Adjust the magnitude as needed

                                    # Transform the movement vector to world space using the entity's transformation matrix
                                    movement_vector = entity.getQuat(render).xform(local_movement)

                                    # Update the position with the transformed movement vector
                                    new_position = position + movement_vector
                                    placeholder = self.new_parent_node.attachNewNode(f'Instanceb_{movement_vector}')
                                    placeholder.setPos(new_position)
                                    placeholder.setHpr(rotation)
                                    if int(height) == 0:
                                        instance = self.moveThroughBoxesh.instanceTo(placeholder)
                                    else:
                                        if int(heightdisplay) == 0:
                                            instance = self.moveThroughBoxesh.instanceTo(placeholder)
                                        else:
                                            instance = self.moveThroughBoxesb.instanceTo(placeholder)
                                    


                            if ghost_entry:
                                ghost_pos = ghost_entry.getIntoNodePath().getPos(render)
                                # Example usage with the collision entry

                                desired_name = "Instanceb_LVecBase3f(0, 0, 0)"
                                if self.check_for_instance_name(ghost_entry.getIntoNodePath(), desired_name):
                                    if int(heightdisplay) == 0:
                                        local_movement = Vec3(0, 0, -1)
                                    else:
                                        local_movement = Vec3(0, 0, -int(height))  # Adjust the magnitude as needed

                                    # Transform the movement vector to world space using the entity's transformation matrix
                                    movement_vector = entity.getQuat(render).xform(local_movement)

                                    # Update the position with the transformed movement vector
                                    new_position = position + movement_vector
                                    self.pointer.setPos(new_position[0],new_position[1],new_position[2]+self.zmem)
                                else:
                                    self.pointer.setPos(ghost_pos[0],ghost_pos[1],ghost_pos[2]+self.zmem)
                            self.placefunc()
                        else:
                            self.pointer.setPos(int(xp), int(zp), int(yp))
                            if self.keypointhidetog == True:
                                self.pointer.show()
                            self.placefunc()

    

                    # entity.setHpr(hpr[0],hpr[1],hpr[2]+1)
                if self.key3tog: 
                    pass
                    # hpr=entity.getHpr()

                    # entity.setHpr(hpr[0],hpr[1]+1,hpr[2])
                if self.key4tog: 
                    pass
                    # hpr=entity.getHpr()

                    # entity.setHpr(hpr[0]+1,hpr[1],hpr[2])
                


                # intance=self.objmap.get(str((hmap0)))
                if self.key1tog: 
                    self.play_dig_sound()
                    
                    size = entity.getPythonTag("size")
                    if size is not None:
                        num_turns = random.randint(0, 3)

                        # Rotate the PNMImage
                        if num_turns == 0:
                            flip_x, flip_y, transpose = False, False, False
                        elif num_turns == 1:
                            flip_x, flip_y, transpose = False, True, True
                        elif num_turns == 2:
                            flip_x, flip_y, transpose = True, True, False
                        else:  # num_turns == 3
                            flip_x, flip_y, transpose = True, False, True

                        # Apply the flip
                        self.small_imagetest.flip(flip_x, flip_y, transpose)

                        # Rotate the numpy heightmap

                        self.brushshovel = np.rot90(self.brushshovel, num_turns)
                        # self.brushshovel = self.rotate(self.brushshovel, num_turns)
                        self.process_heightmaps(x, z, xp, zp, scale_factor,self.small_imagetest,self.brushshovel,20,self.regionfile01,False,False,True)
                        offset = 54
                        self.numedits.append(["handbrushes", "mountains", "brushshovel1", num_turns, int(xp),int(zp),20,False,True])

                        sev=self.savesedit.get(str((hmap0)))
                        if sev is None:
                            # if len(self.numedits) >= 10:
                            self.savesedit[str(hmap0)]=self.numedits
                            self.bufferdictbrush.append('1')

                            self.numedits=[]
                            if len(self.numedits) >= 10:
                                self.savesedit=self.save_dict_as_json(self.world_name_n,self.savesedit,self.bufferdictbrush) 
                        else:
                            # if len(self.numedits) >= 10:
                            for i in self.numedits:
                                self.savesedit[str(hmap0)].append(i)#.insert(0, i)#.append(i)
                                self.bufferdictbrush.append('1')

                            self.numedits=[]
                            if len(self.numedits) >= 10:
                                self.savesedit=self.save_dict_as_json(self.world_name_n,self.savesedit,self.bufferdictbrush) 

                        heightmapsgot = self.height_maps.get((hmap0))
                        txture = entity.getPythonTag("txtu")
                        entity.removeNode()
                        terrain_s3=self.generate_terrainCollide(heightmapsgot, size, Vec3(x, z, 0), 3, txture, hmap0)
                        coord_tuplevi = (x, z)
                        self.checkdic[coord_tuplevi] = terrain_s3

                        # Get the entities at the desired coordinates
                        entity_x_positive = self.checkdic.get((x + offset, z))#left or right
                        entity_x_negative = self.checkdic.get((x - offset, z))#left or right
                        entity_z_positive = self.checkdic.get((x, z + offset))#up or down
                        entity_z_negative = self.checkdic.get((x, z - offset))#up or down

                        entity_top_right = self.checkdic.get((x + offset, z + offset))
                        entity_top_left = self.checkdic.get((x - offset, z + offset))
                        entity_bottom_right = self.checkdic.get((x + offset, z - offset))
                        entity_bottom_left = self.checkdic.get((x - offset, z - offset))

                        if entity_x_positive is not None:
                            txture = entity_x_positive.getPythonTag("txtu")
                            hmap = entity_x_positive.getPythonTag("mappos")
                            heightmapsgot = self.height_maps.get(hmap)

                            entity_x_positive.removeNode()
                            terrain_s3=self.generate_terrainCollide(heightmapsgot, size, Vec3((x + offset, z), 0), 3, txture, hmap)
                            coord_tuplevi = ((x + offset, z))
                            self.checkdic[coord_tuplevi] = terrain_s3
                                # terrain_s3.setColor(0, 0, 1, 1)
                        if entity_x_negative is not None:
                            txture = entity_x_negative.getPythonTag("txtu")
                            hmap = entity_x_negative.getPythonTag("mappos")
                            heightmapsgot = self.height_maps.get(hmap)

                            entity_x_negative.removeNode()
                            terrain_s3=self.generate_terrainCollide(heightmapsgot, size, Vec3((x - offset, z), 0), 3, txture, hmap)
                            coord_tuplevi = ((x - offset, z))
                            self.checkdic[coord_tuplevi] = terrain_s3
                            # terrain_s3.setColor(0, 0, 1, 1)
                        if entity_z_positive is not None:
                            txture = entity_z_positive.getPythonTag("txtu")
                            hmap = entity_z_positive.getPythonTag("mappos")
                            heightmapsgot = self.height_maps.get(hmap)

                            entity_z_positive.removeNode()
                            terrain_s3=self.generate_terrainCollide(heightmapsgot, size, Vec3((x, z + offset), 0), 3, txture, hmap)
                            coord_tuplevi = ((x, z + offset))
                            self.checkdic[coord_tuplevi] = terrain_s3
                        #     # terrain_s3.setColor(0, 0, 1, 1)
                        if entity_z_negative is not None:
                            txture = entity_z_negative.getPythonTag("txtu")
                            hmap = entity_z_negative.getPythonTag("mappos")
                            heightmapsgot = self.height_maps.get(hmap)

                            entity_z_negative.removeNode()
                            terrain_s3=self.generate_terrainCollide(heightmapsgot, size, Vec3((x, z - offset), 0), 3, txture, hmap)
                            coord_tuplevi = ((x, z - offset))
                            self.checkdic[coord_tuplevi] = terrain_s3
                        #     terrain_s3.setColor(0, 0, 1, 1)
                        if entity_top_right is not None:
                            txture = entity_top_right.getPythonTag("txtu")
                            hmap = entity_top_right.getPythonTag("mappos")
                            heightmapsgot = self.height_maps.get(hmap)

                            entity_top_right.removeNode()
                            terrain_s3=self.generate_terrainCollide(heightmapsgot, size, Vec3((x + offset, z + offset), 0), 3, txture, hmap)
                            coord_tuplevi = ((x + offset, z + offset))
                            self.checkdic[coord_tuplevi] = terrain_s3
                        #     # terrain_s3.setColor(0, 0, 1, 1)
                        if entity_top_left is not None:
                            txture = entity_top_left.getPythonTag("txtu")
                            hmap = entity_top_left.getPythonTag("mappos")
                            heightmapsgot = self.height_maps.get(hmap)

                            entity_top_left.removeNode()
                            terrain_s3=self.generate_terrainCollide(heightmapsgot, size, Vec3((x - offset, z + offset), 0), 3, txture, hmap)
                            coord_tuplevi = ((x - offset, z + offset))
                            self.checkdic[coord_tuplevi] = terrain_s3
                        #     # terrain_s3.setColor(0, 0, 1, 1)
                        if entity_bottom_right is not None:
                            txture = entity_bottom_right.getPythonTag("txtu")
                            hmap = entity_bottom_right.getPythonTag("mappos")
                            heightmapsgot = self.height_maps.get(hmap)

                            entity_bottom_right.removeNode()
                            terrain_s3=self.generate_terrainCollide(heightmapsgot, size, Vec3((x + offset, z - offset), 0), 3, txture, hmap)
                            coord_tuplevi = ((x + offset, z - offset))
                            self.checkdic[coord_tuplevi] = terrain_s3
                        #     # terrain_s3.setColor(0, 0, 1, 1)
                        if entity_bottom_left is not None:
                            txture = entity_bottom_left.getPythonTag("txtu")
                            hmap = entity_bottom_left.getPythonTag("mappos")
                            heightmapsgot = self.height_maps.get(hmap)

                            entity_bottom_left.removeNode()
                            terrain_s3=self.generate_terrainCollide(heightmapsgot, size, Vec3((x - offset, z - offset), 0), 3, txture, hmap)
                            coord_tuplevi = ((x - offset, z - offset))
                            self.checkdic[coord_tuplevi] = terrain_s3



    # Define a function to perform the raycast
    def performRaycastLeft(self):
        if base.mouseWatcherNode.hasMouse():
 
            # Get the player's position
            # player_pos = self.player.main_node.getPos()
            # player_pos[2]+= self.player.getConfig("player_height")
            # Get the mouse position in 2D screen space
            mpos = base.mouseWatcherNode.getMouse()

            # Convert the 2D screen space coordinates to 3D ray
            pFrom = Point3()
            pTo = Point3()
            base.camLens.extrude(mpos, pFrom, pTo)

            # Convert the points from camera space to world space
            world_pFrom = render.getRelativePoint(base.cam, pFrom)
            world_pTo = render.getRelativePoint(base.cam, pTo)

            # Calculate the direction vector from the player's position to the mouse position
            direction = world_pTo - world_pFrom
            camera_position = base.camera.getPos(render)
            # Set the origin and direction of the ray
            self.ray.setOrigin(camera_position)
            self.ray.setDirection(direction)

            # Perform the raycast
            self.traverser.traverse(render)
            if self.queue.getNumEntries() > 0:
                self.queue.sortEntries()
                hit = self.queue.getEntry(0)

                hitNodePath = hit.getIntoNodePath()
                hitPos = hit.getSurfacePoint(render)
                entity = hitNodePath.getParent()
                xp,zp,yp=hitPos
                xe,ze,ye= entity.getPos()
                xer,zer,yer= entity.getPos(render)
                ehpr=entity.getHpr(render)
                escale=entity.getScale(render)

                x,z = int(xe),int(ze)
                # Example usage


                hmap0=(int(x  // 486) * 486,int(z // 486) * 486)

                # hmap1=(int(x  // 162) * 162,int(z // 162) * 162)

                # hpr = entity.getHpr()


                # intance=self.objmap.get(str((hmap0)))
                # if self.key2tog: 
                #     hpr=entity.getHpr()

                #     entity.setHpr(hpr[0],hpr[1],hpr[2]-1)
                # if self.key3tog: 
                #     hpr=entity.getHpr()

                #     entity.setHpr(hpr[0],hpr[1]-1,hpr[2])
                # if self.key4tog: 
                #     hpr=entity.getHpr()

                #     entity.setHpr(hpr[0]-1,hpr[1],hpr[2])
                if self.key2tog:
                    if entity.hasTag('coordinate_location'):
                        hasplaced=False
                        pos = entity.getTag('coordinate_location')
                        parts = pos.split('_')
                        if entity.hasTag('placed'):
                            hasplaced=True

                        self.play_chop_sound()
                        entity.removeNode()

                        base_x_keyb = int(xp // 162) * 162
                        base_z_keyb = int(zp // 162) * 162
                        offsets = [-324, -162, 0, 162, 324]

                        for offset_xb in offsets:
                            for offset_zb in offsets:
                                new_x_keyb = base_x_keyb + offset_xb
                                new_z_keyb = base_z_keyb + offset_zb
                                group_key = (new_x_keyb, new_z_keyb)
                                combined_key = f"{'_'.join(parts[4:10])}{group_key}"
                                instances_pos = self.instance_groups.get(combined_key)

                                if instances_pos is not None:
                                    new_instances_pos = [[], [], []]
                                    for pos, rotation, scale in zip(instances_pos[0], instances_pos[1], instances_pos[2]):
                                        entb=self.instance_groups[combined_key][3].get(str([xer,zer,yer])+str([ehpr[0],ehpr[1],ehpr[2]])+str([escale[0],escale[1],escale[2]]))
                                        if entb is not None:
                                            if [pos, rotation, scale] == entb[0:3]:
                                                if combined_key not in self.forremoval:
                                                    self.forremoval[combined_key] = []

                                                if hasplaced == True:

                                                    for key, value in self.forplacement.items():
                                                        positions0, rotations0, scales0, names0 = value
                                                        indices_to_remove = []
                                                        for i, (posb, rotb, scaleb, nameb) in enumerate(zip(positions0, rotations0, scales0, names0)):
                                                            if posb == pos and rotb == rotation and scaleb == scale and nameb == ['_'.join(parts[4:10])]:
                                                                indices_to_remove.append(i)
                                                                self.bufferdict3.append('1')
                                                                
                                                        # Remove matching items by their indices in reverse order to avoid index shifting issues
                                                        for i in sorted(indices_to_remove, reverse=True):
                                                            del positions0[i]
                                                            del rotations0[i]
                                                            del scales0[i]
                                                            del names0[i]
                                                            


                                                else:
                                                    if pos not in self.forremoval[combined_key]:
                                                        self.forremoval[combined_key].append([pos, rotation, scale])
                                                        self.bufferdict2.append('1')

                                                self.forremoval = self.save_dict_as_json(self.world_name_r,self.forremoval,self.bufferdict2)
                                                self.forplacement = self.save_dict_as_json(self.world_name_p,self.forplacement,self.bufferdict3)
                                            else:
                                                new_instances_pos[0].append(pos)
                                                new_instances_pos[1].append(rotation)
                                                new_instances_pos[2].append(scale)
                                        else:
                                            new_instances_pos[0].append(pos)
                                            new_instances_pos[1].append(rotation)
                                            new_instances_pos[2].append(scale)
                                    self.instance_groups[combined_key][0] = new_instances_pos[0]
                                    self.instance_groups[combined_key][1] = new_instances_pos[1]
                                    self.instance_groups[combined_key][2] = new_instances_pos[2]
                                    #     #radius search
                                    #     instance_position = LPoint3f(*pos)
                                    #     distance_from_coordinate_sq = (instance_position - coordinate_location).lengthSquared()
                                    #     if distance_from_coordinate_sq <= (1**2):
  
                                    #         if combined_key not in self.forremoval:
                                    #             self.forremoval[combined_key] = []

                                    #         if [*pos] not in self.forremoval[combined_key]:
                                    #             self.forremoval[combined_key].append([*pos])
                                    #             self.bufferdict2.append('1')
                                    #         else:
                                    #             print('is in')

                                    #         self.forremoval = self.save_dict_as_json('removeddata.json',self.forremoval,self.bufferdict2)
                                    #     else:
                                    #         new_instances_pos[0].append(pos)
                                    #         new_instances_pos[1].append(rotation)
                                    #         new_instances_pos[2].append(scale)
                                    # self.instance_groups[combined_key] = new_instances_pos
                    else:
                        pass

                    # entity.setHpr(hpr[0],hpr[1],hpr[2]-1)
                if self.key3tog: 
                    pass
                    # hpr=entity.getHpr()

                    # entity.setHpr(hpr[0],hpr[1]-1,hpr[2])
                if self.key4tog: 
                    pass
                    # hpr=entity.getHpr()

                    # entity.setHpr(hpr[0]-1,hpr[1],hpr[2])

                if self.key1tog: 
                    self.play_dig_sound()
                    size = entity.getPythonTag("size")
                    if size is not None:
                        num_turns = random.randint(0, 3)

                        # Rotate the PNMImage
                        if num_turns == 0:
                            flip_x, flip_y, transpose = False, False, False
                        elif num_turns == 1:
                            flip_x, flip_y, transpose = False, True, True
                        elif num_turns == 2:
                            flip_x, flip_y, transpose = True, True, False
                        else:  # num_turns == 3
                            flip_x, flip_y, transpose = True, False, True

                        # Apply the flip
                        self.small_imagetest.flip(flip_x, flip_y, transpose)

                        # Rotate the numpy heightmap

                        self.brushshovel = np.rot90(self.brushshovel, num_turns)
                        # self.brushshovel = self.rotate(self.brushshovel, num_turns)
                        self.process_heightmaps(x, z, xp, zp, scale_factor,self.small_imagetest,self.brushshovel,50,self.regionfile01,False,False,False)
                        offset = 54
                        self.numedits.append(["handbrushes","mountains", "brushshovel1", num_turns, int(xp),int(zp),50,False,False])

                        sev=self.savesedit.get(str((hmap0)))
                        if sev is None:
                            # if len(self.numedits) >= 10:
                            self.savesedit[str(hmap0)]=self.numedits
                            self.bufferdictbrush.append('1')

                            self.numedits=[]
                            if len(self.numedits) >= 10:
                                self.savesedit=self.save_dict_as_json(self.world_name_n,self.savesedit,self.bufferdictbrush) 
                        else:
                            # if len(self.numedits) >= 10:
                            for i in self.numedits:
                                self.savesedit[str(hmap0)].append(i)#.insert(0, i)#.append(i)
                                self.bufferdictbrush.append('1')

                            self.numedits=[]
                            if len(self.numedits) >= 10:
                                self.savesedit=self.save_dict_as_json(self.world_name_n,self.savesedit,self.bufferdictbrush) 

                        heightmapsgot = self.height_maps.get((hmap0))
                        txture = entity.getPythonTag("txtu")
                        entity.removeNode()
                        terrain_s3=self.generate_terrainCollide(heightmapsgot, size, Vec3(x, z, 0), 3, txture, hmap0)
                        coord_tuplevi = (x, z)
                        self.checkdic[coord_tuplevi] = terrain_s3

                        # Get the entities at the desired coordinates
                        entity_x_positive = self.checkdic.get((x + offset, z))#left or right
                        entity_x_negative = self.checkdic.get((x - offset, z))#left or right
                        entity_z_positive = self.checkdic.get((x, z + offset))#up or down
                        entity_z_negative = self.checkdic.get((x, z - offset))#up or down

                        entity_top_right = self.checkdic.get((x + offset, z + offset))
                        entity_top_left = self.checkdic.get((x - offset, z + offset))
                        entity_bottom_right = self.checkdic.get((x + offset, z - offset))
                        entity_bottom_left = self.checkdic.get((x - offset, z - offset))

                        if entity_x_positive is not None:
                            txture = entity_x_positive.getPythonTag("txtu")
                            hmap = entity_x_positive.getPythonTag("mappos")
                            heightmapsgot = self.height_maps.get(hmap)

                            entity_x_positive.removeNode()
                            terrain_s3=self.generate_terrainCollide(heightmapsgot, size, Vec3((x + offset, z), 0), 3, txture, hmap)
                            coord_tuplevi = ((x + offset, z))
                            self.checkdic[coord_tuplevi] = terrain_s3
                                # terrain_s3.setColor(0, 0, 1, 1)
                        if entity_x_negative is not None:
                            txture = entity_x_negative.getPythonTag("txtu")
                            hmap = entity_x_negative.getPythonTag("mappos")
                            heightmapsgot = self.height_maps.get(hmap)

                            entity_x_negative.removeNode()
                            terrain_s3=self.generate_terrainCollide(heightmapsgot, size, Vec3((x - offset, z), 0), 3, txture, hmap)
                            coord_tuplevi = ((x - offset, z))
                            self.checkdic[coord_tuplevi] = terrain_s3
                            # terrain_s3.setColor(0, 0, 1, 1)
                        if entity_z_positive is not None:
                            txture = entity_z_positive.getPythonTag("txtu")
                            hmap = entity_z_positive.getPythonTag("mappos")
                            heightmapsgot = self.height_maps.get(hmap)

                            entity_z_positive.removeNode()
                            terrain_s3=self.generate_terrainCollide(heightmapsgot, size, Vec3((x, z + offset), 0), 3, txture, hmap)
                            coord_tuplevi = ((x, z + offset))
                            self.checkdic[coord_tuplevi] = terrain_s3
                        #     # terrain_s3.setColor(0, 0, 1, 1)
                        if entity_z_negative is not None:
                            txture = entity_z_negative.getPythonTag("txtu")
                            hmap = entity_z_negative.getPythonTag("mappos")
                            heightmapsgot = self.height_maps.get(hmap)

                            entity_z_negative.removeNode()
                            terrain_s3=self.generate_terrainCollide(heightmapsgot, size, Vec3((x, z - offset), 0), 3, txture, hmap)
                            coord_tuplevi = ((x, z - offset))
                            self.checkdic[coord_tuplevi] = terrain_s3
                        #     terrain_s3.setColor(0, 0, 1, 1)
                        if entity_top_right is not None:
                            txture = entity_top_right.getPythonTag("txtu")
                            hmap = entity_top_right.getPythonTag("mappos")
                            heightmapsgot = self.height_maps.get(hmap)

                            entity_top_right.removeNode()
                            terrain_s3=self.generate_terrainCollide(heightmapsgot, size, Vec3((x + offset, z + offset), 0), 3, txture, hmap)
                            coord_tuplevi = ((x + offset, z + offset))
                            self.checkdic[coord_tuplevi] = terrain_s3
                        #     # terrain_s3.setColor(0, 0, 1, 1)
                        if entity_top_left is not None:
                            txture = entity_top_left.getPythonTag("txtu")
                            hmap = entity_top_left.getPythonTag("mappos")
                            heightmapsgot = self.height_maps.get(hmap)

                            entity_top_left.removeNode()
                            terrain_s3=self.generate_terrainCollide(heightmapsgot, size, Vec3((x - offset, z + offset), 0), 3, txture, hmap)
                            coord_tuplevi = ((x - offset, z + offset))
                            self.checkdic[coord_tuplevi] = terrain_s3
                        #     # terrain_s3.setColor(0, 0, 1, 1)
                        if entity_bottom_right is not None:
                            txture = entity_bottom_right.getPythonTag("txtu")
                            hmap = entity_bottom_right.getPythonTag("mappos")
                            heightmapsgot = self.height_maps.get(hmap)

                            entity_bottom_right.removeNode()
                            terrain_s3=self.generate_terrainCollide(heightmapsgot, size, Vec3((x + offset, z - offset), 0), 3, txture, hmap)
                            coord_tuplevi = ((x + offset, z - offset))
                            self.checkdic[coord_tuplevi] = terrain_s3
                        #     # terrain_s3.setColor(0, 0, 1, 1)
                        if entity_bottom_left is not None:
                            txture = entity_bottom_left.getPythonTag("txtu")
                            hmap = entity_bottom_left.getPythonTag("mappos")
                            heightmapsgot = self.height_maps.get(hmap)

                            entity_bottom_left.removeNode()
                            terrain_s3=self.generate_terrainCollide(heightmapsgot, size, Vec3((x - offset, z - offset), 0), 3, txture, hmap)
                            coord_tuplevi = ((x - offset, z - offset))
                            self.checkdic[coord_tuplevi] = terrain_s3
                    #     # terrain_s3.setColor(0, 0, 1, 1)

                        # heightmap_key = (int((positen[0] // 486) * 487) , int((positen[1] // 486) * 487))
                        # heightmapsgot = self.height_maps.get(heightmap_key)
                        # entity.removeNode()
                        # terrain_s3=self.generate_terrainCollide(heightmapsgot, size, Vec3(xe, ye, 0), 3, txture, True)
                        # coord_tuplevi = (xe, ye)
                        # self.checkdic[coord_tuplevi] = terrain_s3

                        # bigmesh.removeNode()
                        # terrain_s3=self.generate_terrain(heightmapsgot, 28, Vec3(new_x, new_z, -0.1), 6, txture)
                        # coord_tuplevi = (new_x, new_z)
                        # self.dictn[coord_tuplevi] = terrain_s3


app = MyApp()
app.run()