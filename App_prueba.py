import streamlit as st
import SimpleITK as sitk
import gdown
import tempfile
import os
import requests
import textwrap
from PIL import Image

#### Funciones propias #### 
from RotarVolumen import leer_archivos_dicom_mult, process_dicom_mult
from inferencia import uso_RUBEN_mult
from conversor import carpetaPNG, carpetaDCM

#### Estilo HTML ####

#### Tarjeta #### 
st.markdown('''
        <style>
        .card {
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 12px;
            border: 1px solid #ddd;
            margin-top: 10px;
        }
        .card h4 {
            margin-top: 0;
        }
        </style>
        ''', unsafe_allow_html=True)

#### Inicializar #### 
if 'screen' not in st.session_state:
    st.session_state.screen = 1

def go_welcome():
    st.session_state.screen = 1

def go_lobby():
    st.session_state.screen = 2

#### Presentacion #### 
if st.session_state.screen == 1:
    st.title('Welcome')
    st.markdown('Press **Start** to enter')
    st.button('Start ▶️', on_click = go_lobby)
	
#### Lobby #### 
elif st.session_state.screen == 2:

    st.set_page_config(layout='wide')
	
    upload_dcm_1 = st.sidebar.file_uploader('**Archivos DCM 1**', type=['DCM'],
                                            accept_multiple_files=True, key='dcm1')
    
    upload_dcm_2 = st.sidebar.file_uploader('**Archivos DCM 2**', type=['DCM'],
                                            accept_multiple_files=True, key='dcm2')

    upload_dcm_3 = st.sidebar.file_uploader('**Archivos DCM 3**', type=['DCM'],
                                            accept_multiple_files=True, key='dcm3')

    upload_dcm_4 = st.sidebar.file_uploader('**Archivos DCM 4**', type=['DCM'],
                                            accept_multiple_files=True, key='dcm4')

    upload_dcm_5 = st.sidebar.file_uploader('**Archivos DCM 5**', type=['DCM'],
                                            accept_multiple_files=True, key='dcm5')

    upload_dcm_6 = st.sidebar.file_uploader('**Archivos DCM 6**', type=['DCM'],
                                            accept_multiple_files=True, key='dcm6')
    
    upload_dcm_7 = st.sidebar.file_uploader('**Archivos DCM 7**', type=['DCM'],
                                            accept_multiple_files=True, key='dcm7')

    upload_dcm_8 = st.sidebar.file_uploader('**Archivos DCM 8**', type=['DCM'],
                                            accept_multiple_files=True, key='dcm8')

    upload_dcm_9 = st.sidebar.file_uploader('**Archivos DCM 9**', type=['DCM'],
                                            accept_multiple_files=True, key='dcm9')

    upload_dcm_10 = st.sidebar.file_uploader('**Archivos DCM 10**', type=['DCM'],
                                            accept_multiple_files=True, key='dcm10')

    upload_dcm_11 = st.sidebar.file_uploader('**Archivos DCM 11**', type=['DCM'],
                                            accept_multiple_files=True, key='dcm11')
    
    if upload_dcm_1:
        
        upload_dcms = [upload_dcm_1,upload_dcm_2,upload_dcm_3]#,upload_dcm_4,upload_dcm_5,
                       #upload_dcm_6,upload_dcm_7,upload_dcm_8,upload_dcm_9,upload_dcm_10,
                       #upload_dcm_11]
        
        st.write(len(upload_dcms))
        # ------------------------------------------------------------------------------------
        #                                   SUBIDA DE DATOS
        # ------------------------------------------------------------------------------------
		#### Carpeta temporal DICOM ####
        if 'rutas_DCM' not in st.session_state:
            rutas_DCM = []

            for upload_dcm in upload_dcms:
                temp_dcm_org = tempfile.mkdtemp()
                
                for f in upload_dcm:
                    path = os.path.join(temp_dcm_org, f.name)
                    with open(path, 'wb') as out:
                        out.write(f.getbuffer())

                rutas_DCM.append(temp_dcm_org)

            st.session_state.rutas_DCM = rutas_DCM

        rutas_DCM = st.session_state.rutas_DCM

        # ------------------------------------------------------------------------------------
        #                                      ORIGINAL
        # ------------------------------------------------------------------------------------

        #### Volumenes ####
        if 'HV_org' not in st.session_state:
            st.session_state.HV_org = leer_archivos_dicom_mult(rutas_DCM)

        HV_org = st.session_state.HV_org

		#### Carpetas temporal PNG ####
        if 'temp_png_orgs' not in st.session_state:
            
            st.session_state.temp_png_orgs = [carpetaPNG(V_org,0) for V_org in HV_org]

        temp_png_orgs = st.session_state.temp_png_orgs

        # ------------------------------------------------------------------------------------
        #                                   ESTÁNDAR
        # ------------------------------------------------------------------------------------
        #### Inferencia de los parámetros ####
        if 'p_std' not in st.session_state: 
            url_estandar = 'https://drive.google.com/uc?export=download&id=10BglUsjZLKeeiqpGG5xY-Isy8vGAzrNg'
            output_estandar = 'modelo_estandar.pt'
            gdown.download(url_estandar, output_estandar, quiet=False)
            st.session_state.p_std = uso_RUBEN_mult('modelo_estandar.pt',
                                                    temp_png_orgs)
        
        p_std = st.session_state.p_std

    	#### Rotación ####
        if 'HV_std' not in st.session_state:
            HV_std, spcs_std = process_dicom_mult(p_std,rutas_DCM)

            st.session_state.HV_std = HV_std
            st.session_state.spcs_std = spcs_std

        spcs_std = st.session_state.spcs_std
        HV_std = st.session_state.HV_std
		
		#### Carpeta temporal DICOM ####
        if 'temp_dcm_stds' not in st.session_state:		

            st.session_state.temp_dcm_stds = [carpetaDCM(HV_std[i], spcs_std [i]) for i in range(len(HV_std))]

        temp_dcm_stds = st.session_state.temp_dcm_stds
        
		#### Carpeta temporal PNG ####
        if 'temp_png_stds' not in st.session_state:
            
            st.session_state.temp_png_stds  = [carpetaPNG(V_std,0) for V_std in HV_std]
            
        temp_png_stds = st.session_state.temp_png_stds

        # ------------------------------------------------------------------------------------
        #                                         LVOT
        # ------------------------------------------------------------------------------------
        #### Inferencia de los parámetros ####
        if 'p_LVOT' not in st.session_state:
            url_LVOT = 'https://drive.google.com/uc?export=download&id=15g-yPz4mVrDpgN4h0tnuuE7dt7-ZxRj_'
            output_LVOT = 'modelo_LVOT.pt'
            gdown.download(url_LVOT, output_LVOT, quiet=False)
            st.session_state.p_LVOT = uso_RUBEN_mult('modelo_LVOT.pt',
													 temp_png_stds)
        
        p_LVOT = st.session_state.p_LVOT

    	#### Rotación ####
        if 'HV_LVOT' not in st.session_state:
            HV_LVOT, _ = process_dicom_mult(p_LVOT,temp_dcm_stds)

            st.session_state.HV_LVOT = HV_LVOT

        HV_LVOT = st.session_state.HV_LVOT

		#### Carpeta temporal PNG ####
        if 'temp_png_LVOTs' not in st.session_state:
            
            st.session_state.temp_png_LVOTs  = [carpetaPNG(V_LVOT,0) for V_LVOT in HV_LVOT]
            
        temp_png_LVOTs = st.session_state.temp_png_LVOTs

        html_3 = f'''
            <div class="card">
                <h4>Imagenes</h4>
            </div>
        '''
        st.markdown(textwrap.dedent(html_3), unsafe_allow_html=True)
        
        N_org_1 = st.slider('Volumen',min_value=1, max_value=len(HV_org), step=1,key ='N_org_1')
        N_org_2 = st.slider('Volumen',min_value=1, max_value=HV_org[0].shape[0], step=1,key ='N_org_2')

    	#### Rotación ####
        if 'HV_LVOT' not in st.session_state:
            HV_LVOT, _ = process_dicom_mult(p_LVOT,temp_dcm_stds)

            st.session_state.HV_LVOT = HV_LVOT

        HV_LVOT = st.session_state.HV_LVOT

		#### Carpeta temporal PNG ####
        if 'temp_png_LVOTs' not in st.session_state:
            
            st.session_state.temp_png_LVOTs  = [carpetaPNG(V_LVOT,0) for V_LVOT in HV_LVOT]
            
        temp_png_LVOTs = st.session_state.temp_png_LVOTs

        html_3 = f'''
            <div class="card">
                <h4>Imagenes</h4>
            </div>
        '''
        st.markdown(textwrap.dedent(html_3), unsafe_allow_html=True)
        
        N_org_1 = st.slider('Volumen',min_value=1, max_value=len(HV_org), step=1,key ='N_org1')
        N_org_2 = st.slider('Volumen',min_value=1, max_value=HV_org[0].shape[0], step=1,key ='N_org2')

        img_orig_user_1 = Image.open(os.path.join(temp_png_orgs[N_org_1-1], f'slice_{(N_org_2-1):03d}.png'))
        img_orig_user_2 = Image.open(os.path.join(temp_png_stds[N_org_1-1], f'slice_{(N_org_2-1):03d}.png'))

        st.image(img_orig_user_1, caption='Original', use_container_width=False)
        st.image(img_orig_user_2, caption='Original', use_container_width=False)
