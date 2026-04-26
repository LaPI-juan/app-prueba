import SimpleITK as sitk
import matplotlib.pyplot as plt 
import numpy as np
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import uuid
import zipfile
import io
import torch
import torchvision as tv
import torch.nn as nn
from ultralytics import YOLO

# ------------------------------------------------------------------------------------
#                                   Leer volumen DICOM
# ------------------------------------------------------------------------------------
def leer_archivos_dicom(dicom_folder):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder)
    reader.SetFileNames(dicom_names)
    volumen = reader.Execute()
    return volumen

# ------------------------------------------------------------------------------------
#                             Remuestreo a espaciado definido
# ------------------------------------------------------------------------------------
def remuestrear_volumen(volumen, new_spacing, new_size=None, interpolador=sitk.sitkBSpline):#interpolador=sitk.sitkLinear):
    original_spacing = np.array(volumen.GetSpacing())
    original_size = np.array(volumen.GetSize())
    
    if new_size is None:
        new_size = [
            int(round(original_size[i] * (original_spacing[i] / new_spacing[i])))
            for i in range(3)
        ]
        
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interpolador)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize([int(s) for s in new_size])
    resampler.SetOutputOrigin(volumen.GetOrigin())
    resampler.SetOutputDirection(volumen.GetDirection())
    resampler.SetDefaultPixelValue(0)

    return resampler.Execute(volumen)

# ------------------------------------------------------------------------------------
#                      Rotación 3D alrededor de un eje arbitrario
# ------------------------------------------------------------------------------------
def aplicar_rotacion(volumen, axis, angulo_grados):
    axis = np.array(axis, dtype=float) * -1
    axis /= np.linalg.norm(axis)
    
    angulo_radianes = np.deg2rad(angulo_grados)
    transform = sitk.VersorRigid3DTransform()
    transform.SetRotation(axis.tolist(), angulo_radianes)
    
    centro = np.array(volumen.TransformContinuousIndexToPhysicalPoint(np.array(volumen.GetSize())/2.0))
    transform.SetCenter(centro.tolist())
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(volumen)
    resampler.SetInterpolator(sitk.sitkBSpline) # Interpolador lineal 
    #resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(transform)
    resampler.SetDefaultPixelValue(0)
    
    return resampler.Execute(volumen)

# ------------------------------------------------------------------------------------
#                         Medir distancia real entre dos puntos
# ------------------------------------------------------------------------------------
def medir_distancia(volumen, p1, p2):
    p1_phys = np.array(volumen.TransformIndexToPhysicalPoint(p1))
    p2_phys = np.array(volumen.TransformIndexToPhysicalPoint(p2))
    dist = np.linalg.norm(p2_phys - p1_phys)
    return dist

# ------------------------------------------------------------------------------------
#                                  Pipeline Completo
# ------------------------------------------------------------------------------------
def process_dicom(dicom_folder, axis, angulo, medir=False):
    
    volumen_original = leer_archivos_dicom(dicom_folder)
    
    volumen_iso = remuestrear_volumen(volumen_original, new_spacing=(1.0, 1.0, 1.0))
    
    volumen_rot =  aplicar_rotacion(volumen_iso, axis, angulo)
    
    volumen_final = remuestrear_volumen(volumen_rot,
                                        new_spacing=volumen_original.GetSpacing(),
                                        new_size=volumen_original.GetSize())
    
    if medir:
        p1 = (20, 20, 20)
        p2 = (43, 43, 43)
        dist_orig = medir_distancia(volumen_original, p1, p2)
        dist_iso = medir_distancia(volumen_iso, p1, p2)
        dist_rot = medir_distancia(volumen_rot, p1, p2)
        dist_final = medir_distancia(volumen_final, p1, p2)
		
    return volumen_final

# ------------------------------------------------------------------------------------
#                                 Funciones multiples
# ------------------------------------------------------------------------------------
def leer_archivos_dicom_mult(rutas_DCM):
    vols_dcm = []
    for ruta in rutas_DCM:
        vol_dcm = leer_archivos_dicom(ruta)
        vol_array = sitk.GetArrayFromImage(vol_dcm)
        vols_dcm.append(vol_array)

    return vols_dcm

def process_dicom_mult(parametros, rutas_DCM):
    vols_fnl, spacings = [], []  

    for i in range(len(rutas_DCM)):
        img_fnl = process_dicom(rutas_DCM[i],
                                         axis=[parametros[i][1],parametros[i][2],parametros[i][3]],
                                         angulo=parametros[i][0],
                                         medir=True)
        
        VF = sitk.GetArrayFromImage(img_fnl)
        VF_spacing = img_fnl.GetSpacing()
        vols_fnl.append(VF)
        spacings.append(VF_spacing)

    return vols_fnl, spacings

# ------------------------------------------------------------------------------------
#                                    Carpeta PNG
# ------------------------------------------------------------------------------------
def img_slc(Vol,img_mode,SliceNum):

    if img_mode == 0:
        Vol_raw = Vol[SliceNum,:,:]
        Vol_norm = (Vol_raw-np.min(Vol_raw))/(np.max(Vol_raw)-np.min(Vol_raw))
        Vol_img = (Vol_norm*255).astype(np.uint8)
        img_user = Image.fromarray(Vol_img,mode='L')
		
    else:
        Vol_raw = Vol[SliceNum,:,:,:]
        Vol_norm = (Vol_raw-np.min(Vol_raw))/(np.max(Vol_raw)-np.min(Vol_raw))
        Vol_img = (Vol_norm*255).astype(np.uint8)
        img_user = Image.fromarray(Vol_img,mode='RGB')  

    return img_user

def carpetaPNG(Vol,img_mode):
    Vol = np.array(Vol)

    temp_png = tempfile.mkdtemp()
    png_paths = []

    for i in range(Vol.shape[0]):
        img_png = img_slc(Vol,img_mode,i)
        path = os.path.join(temp_png, f'slice_{i:03d}.png')
        img_png.save(path)
        png_paths.append(path)	

    return temp_png

# ------------------------------------------------------------------------------------
#                                    Carpeta DCM
# ------------------------------------------------------------------------------------
def uid():
    return '2.25.' + str(int(uuid.uuid4()))

def carpetaDCM(volume_np, spacing=(1, 1, 1),
                           modality='CT',
                           patient_name='Paciente',
                           patient_id='0000'):
    
    temp_dir_dicom = tempfile.mkdtemp()
    dicom_paths = []

    volume_np = volume_np.astype(np.int16)

    img3d = sitk.GetImageFromArray(volume_np)
    img3d.SetSpacing(spacing)

    study_uid = uid()
    series_uid = uid()

    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()

    array = sitk.GetArrayFromImage(img3d)

    for z in range(array.shape[0]):
        slice_np = array[z, :, :]

        slice_img = sitk.GetImageFromArray(slice_np[None, :, :])
        slice_img.SetSpacing(spacing)

        slice_img.SetMetaData('0008|0060', modality)
        slice_img.SetMetaData('0010|0010', patient_name)
        slice_img.SetMetaData('0010|0020', patient_id)

        slice_img.SetMetaData('0020|000D', study_uid)
        slice_img.SetMetaData('0020|000E', series_uid)
        slice_img.SetMetaData('0008|0018', uid())

        slice_img.SetMetaData('0020|0013', str(z + 1))
        slice_img.SetMetaData('0020|0032', f'0\\0\\{z*spacing[2]}')
        slice_img.SetMetaData('0020|0037', '1\\0\\0\\0\\1\\0')

        slice_img.SetMetaData('0018|0050', str(spacing[2]))

        out_path = os.path.join(temp_dir_dicom, f'slice_{z:03d}.dcm')
        writer.SetFileName(out_path)
        writer.Execute(slice_img)

        dicom_paths.append(out_path)

    return temp_dir_dicom

# ------------------------------------------------------------------------------------
#                                   Descarga PNG
# ------------------------------------------------------------------------------------
def descargaPNG(folder_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                path = os.path.join(root, file)
                arcname = os.path.relpath(path, folder_path)
                zipf.write(path, arcname)

    buffer.seek(0)
    return buffer

# ------------------------------------------------------------------------------------
#                                    Resize3D
# ------------------------------------------------------------------------------------
class Resize3D:
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, volume):
        if not isinstance(volume, torch.Tensor):
            volume = torch.tensor(volume, dtype=torch.float32)
        resized_volume = nn.functional.interpolate(volume.unsqueeze(0).unsqueeze(0), size=self.target_size, mode='trilinear')
        volumenFinal =  resized_volume.squeeze(0)
        
        return volumenFinal

# ------------------------------------------------------------------------------------
#                                 Inferencia RUBEN
# ------------------------------------------------------------------------------------
def CargarVolumen(root):
    volumeName = root
    sliceFiles = sorted(os.listdir(volumeName))
    volumeSlices = []
    for slice_file in sliceFiles:
        img_name = os.path.join(volumeName, slice_file)
        image = Image.open(img_name)
        volumeSlices.append(np.array(image))
    volume = np.stack(volumeSlices, axis=0)
    volume = torch.tensor(volume, dtype=torch.float32)
    volume = Resize3D(target_size=(64,112,112))(volume)
    volume = volume / torch.max(volume).item()
    volume = volume.repeat(3, 1, 1, 1).unsqueeze(0)
    return volume

def MViTV2S(weights):
    modelo = tv.models.video.mvit_v2_s()
    modelo.head = nn.Sequential(nn.Linear(in_features=768, out_features=512),
                                nn.Linear(512, 512),
                                nn.Linear(512, 4))
    stateDict = torch.load(weights, map_location=torch.device('cpu'),weights_only=True)
    modelo.load_state_dict(stateDict)
    return modelo

def uso_RUBEN(ruta_modelo, ruta_PNG):

	modelo = MViTV2S(ruta_modelo)
	volumen = CargarVolumen(ruta_PNG)

	modelo.eval()
	with torch.no_grad():
	    salida = modelo(volumen)

	eje = salida[0,1:].cpu().numpy()
	eje = eje / np.linalg.norm(eje)

	ang = float(salida[0][0])
	p_1, p_2, p_3 = float(eje[0]), float(eje[1]), float(eje[2])

	return ang, p_1, p_2, p_3

def uso_RUBEN_mult(ruta_modelo, rutas_PNG):
    parametros = []
    for ruta_PNG in rutas_PNG:
        prm = uso_RUBEN(ruta_modelo,ruta_PNG)
        parametros.append(prm)

    return parametros
# ------------------------------------------------------------------------------------
#                                 Inferencia YOLO
# ------------------------------------------------------------------------------------
def CargarVolumen_YOLO(root):
    volumeName = root
    sliceFiles = sorted(os.listdir(volumeName))
    volumeSlices = []
    for slice_file in sliceFiles:
        img_name = os.path.join(volumeName, slice_file)
        image = Image.open(img_name)
        volumeSlices.append(np.array(image))
    volume = np.stack(volumeSlices, axis=0)
    d, h, w = volume.shape
    volume = torch.tensor(volume, dtype=torch.float32)
    volume = Resize3D(target_size=(d,640,640))(volume)

    volume = volume.squeeze(0).unsqueeze(3).repeat(1, 1, 1, 3).numpy().astype(np.uint8)

    return volume

def uso_YOLO(ruta_modelo, ruta_PNG):
    
    volumen = CargarVolumen_YOLO(ruta_PNG)
    img_list = [vol for vol in volumen]
    
    model = YOLO(ruta_modelo)
    results = [model(img) for img in img_list]
    
    vol_RGB, vol_masks = [], []
    indc, i = [], -1
    
    for r in results:
        conf_arr = r[0].boxes.conf.cpu().numpy()
        if conf_arr.size and conf_arr.max() > 0.65:
            i += 1
            j = int(np.argmax(conf_arr))
            vol_RGB.append(cv2.cvtColor(r[0].plot(),cv2.COLOR_BGR2RGB))
            vol_masks.append(r[0].masks.data.cpu().numpy()[j,:,:])
            
            if r[0].boxes.cls.cpu().numpy()[0] == 0:
                indc.append(i)
            else:
                pass
        else:
            pass
            
    return np.array(vol_RGB), np.array(vol_masks), indc[0], indc[-1]

def uso_YOLO_mult(ruta_modelo, rutas_PNG):
    vols_RGB, vols_masks, indcs = [], [], []

    for ruta_PNG in rutas_PNG:
        vol_RGB, vol_masks, indc_min, indc_max = uso_YOLO(ruta_modelo, ruta_PNG)
        vols_RGB.append(vol_RGB)
        vols_masks.append(vol_masks)
        indcs.append([indc_min, indc_max])

    return vols_RGB, vols_masks, indcs
