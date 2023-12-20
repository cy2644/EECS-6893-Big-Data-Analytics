import scipy.io
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
import numpy as np
import os
import glob

mat_file_path = 'image_data.mat'
mat_contents = scipy.io.loadmat(mat_file_path)
image_data_array = mat_contents['image_data']
output_directory = 'output_dicom_directory'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

def extract_info(struct):
    patient_id = struct['Patient_ID'][0][0]
    image_number = struct['Number'][0][0]
    if isinstance(patient_id, bytes):
        patient_id = patient_id.decode('utf-8')
    elif isinstance(patient_id, np.ndarray):
        if patient_id.dtype.type is np.bytes_:
            patient_id = patient_id.item().decode('utf-8')
        else:
            patient_id = str(patient_id.item())
    if isinstance(image_number, np.ndarray):
        image_number = int(image_number.item())
    else:
        image_number = int(image_number)
    return patient_id, image_number

def create_dicom_file(data, patient_id, image_number, output_path):
    patient_id = str(patient_id)
    image_number = int(image_number)
    file_meta = FileMetaDataset()
    file_meta.FileMetaInformationVersion = b'\x00\x01'
    file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID
    ds = Dataset()
    ds.file_meta = file_meta
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.PatientID = patient_id[:64]
    ds.InstanceNumber = image_number
    ds.PixelData = data.tobytes()
    ds.Rows = data.shape[0]
    ds.Columns = data.shape[1]
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsStored = 16
    ds.BitsAllocated = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    pydicom.dcmwrite(output_path, ds)

for index, struct in enumerate(image_data_array[0]):
    image_data = np.array(struct['Data'], dtype=np.uint16)
    patient_id, image_number = extract_info(struct)
    if not patient_id or not image_number:
        print(f"Skipping index {index} due to invalid patient_id or image_number")
        continue
    output_path = os.path.join(output_directory, f'image_{patient_id}_{image_number}.dcm')
    create_dicom_file(image_data, patient_id, image_number, output_path)

print("DICOM file creation complete.")

new_image_data_struct = []

for index, struct in enumerate(image_data_array[0]):
    patient_id, image_number = extract_info(struct)
    output_path = os.path.join(output_directory, f'image_{patient_id}_{image_number}.dcm')
    try:
        ds = pydicom.dcmread(output_path, force=True)
    except Exception as e:
        print(f"Error reading DICOM file {output_path}: {e}")
        continue
    image_struct = {
        'Patient_ID': np.array([patient_id], dtype='object'),
        'Data': ds.pixel_array,
        'Number': np.array([image_number], dtype='object'),
        'Designator': np.array([struct['Designator'][0]], dtype='object'),
        'Imaging_Type': np.array([struct['Imaging_Type'][0]], dtype='object'),
        'Technique': np.array([struct['Technique'][0]], dtype='object'),
    }
    new_image_data_struct.append(image_struct)

dtypes = [
    ('Patient_ID', 'O'), ('Data', 'O'), ('Number', 'O'), 
    ('Designator', 'O'), ('Imaging_Type', 'O'), ('Technique', 'O')
]
new_image_data_array = np.zeros((len(new_image_data_struct),), dtype=dtypes)
for i, struct in enumerate(new_image_data_struct):
    for field in dtypes:
        new_image_data_array[i][field[0]] = struct[field[0]]

new_mat_file_path = 'new_image_data.mat'
scipy.io.savemat(new_mat_file_path, {'image_data': new_image_data_array})

print("New .mat file with DICOM data created.")
