import streamlit as st
import SimpleITK as sitk
import tempfile
import os
import textwrap
from PIL import Image

#### Funciones propias #### 
from RotarVolumen import leer_archivos_dicom_mult
from conversor import carpetaPNG

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
	
    upload_dcm_1 = st.sidebar.file_uploader('**Archivos DCM**', type=['DCM'],
                                            accept_multiple_files=True, key='dcm1')
    
    upload_dcm_2 = st.sidebar.file_uploader('**Archivos DCM**', type=['DCM'],
                                            accept_multiple_files=True, key='dcm2')

    upload_dcm_3 = st.sidebar.file_uploader('**Archivos DCM**', type=['DCM'],
                                            accept_multiple_files=True, key='dcm3')
    
    if upload_dcm_1:

        # ------------------------------------------------------------------------------------
        #                                      ORIGINAL
        # ------------------------------------------------------------------------------------
		#### Carpeta temporal DICOM ####
        if 'temp_dcm_org_1' not in st.session_state:
            temp_dcm_org_1 = tempfile.mkdtemp()
            for f in upload_dcm_1:
                path = os.path.join(temp_dcm_org_1, f.name)
                with open(path, "wb") as out:
                    out.write(f.getbuffer())

            st.session_state.temp_dcm_org_1 = temp_dcm_org_1

        if 'temp_dcm_org_2' not in st.session_state:
            temp_dcm_org_2 = tempfile.mkdtemp()
            for f in upload_dcm_2:
                path = os.path.join(temp_dcm_org_2, f.name)
                with open(path, "wb") as out:
                    out.write(f.getbuffer())

            st.session_state.temp_dcm_org_2 = temp_dcm_org_2
                 
        if 'temp_dcm_org_3' not in st.session_state:
            temp_dcm_org_3 = tempfile.mkdtemp()
            for f in upload_dcm_3:
                path = os.path.join(temp_dcm_org_3, f.name)
                with open(path, "wb") as out:
                    out.write(f.getbuffer())

            st.session_state.temp_dcm_org_3 = temp_dcm_org_3

        temp_dcm_org_1 = st.session_state.temp_dcm_org_1
        temp_dcm_org_2 = st.session_state.temp_dcm_org_2
        temp_dcm_org_3 = st.session_state.temp_dcm_org_3

        rutas_DCM = [temp_dcm_org_1,temp_dcm_org_2,temp_dcm_org_3]

        #### Volumenes ####
        if 'HV_org' not in st.session_state:
            st.session_state.HV_org = leer_archivos_dicom_mult(rutas_DCM)

        HV_org = st.session_state.HV_org
        
        if 'temp_png_orgs' not in st.session_state:
            
            st.session_state.temp_png_orgs = [carpetaPNG(V_org,0) for V_org in HV_org]

        temp_png_orgs = st.session_state.temp_png_orgs

        html_3 = f'''
            <div class="card">
                <h4>Imagenes</h4>
            </div>
        '''
        st.markdown(textwrap.dedent(html_3), unsafe_allow_html=True)
        
        N_org_1 = st.slider('Volumen',min_value=1, max_value=len(HV_org), step=1,key ='N_org_1')
        N_org_2 = st.slider('Volumen',min_value=1, max_value=HV_org[0].shape[0], step=1,key ='N_org_2')

        img_orig_user_1 = Image.open(os.path.join(temp_png_orgs[N_org_1-1], f'slice_{(N_org_2-1):03d}.png'))

        st.image(img_orig_user_1, caption='Original', use_container_width=False)
