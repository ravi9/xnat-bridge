"""
This is a very basic sample example which does the following:
1. Connect to  XNAT DB
2. Download a sample DICOM Image form XNAT DB
3. Read the DICOM image using pydicom
4. Send the data to OpenVINO model server for inference.
"""

# Connect to XNAT database
import xnat
connection = xnat.connect('https://central.xnat.org', user="rpanchum", password="ravi123")

# Get a sample DICOM image
project = 'Sample_DICOM'
subject = 'dcmtest1'
session = 0
scan = 0
resource = 'DICOM'
file_obj = connection.projects[project].subjects[subject].experiments[session].scans[scan].resources[resource].files[0]

# Read the DICOM image 
import pydicom
with file_obj.open() as fin:
    dataset = pydicom.dcmread(fin)

# Print DICOM Data Info
print()
print("Filename.........:", file_obj.fulluri)
print("Storage type.....:", dataset.SOPClassUID)
print()

pat_name = dataset.PatientName
display_name = pat_name.family_name + ", " + pat_name.given_name
print("Patient's name...:", display_name)
print("Patient id.......:", dataset.PatientID)
print("Modality.........:", dataset.Modality)
print("Study Date.......:", dataset.StudyDate)

if 'PixelData' in dataset:
    rows = int(dataset.Rows)
    cols = int(dataset.Columns)
    print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(
        rows=rows, cols=cols, size=len(dataset.PixelData)))
    if 'PixelSpacing' in dataset:
        print("Pixel spacing....:", dataset.PixelSpacing)

    dicom_image_arr = dataset.pixel_array

# Send the data to OpenVINO model server for inference.
import numpy as np
import json
import requests
import time

#Create a dummy data for UNet, as the dicom image read above does not have 4 channels.
input_data = np.random.randint(0, 255, size=(1, 4, 160,160))

data_obj = {'inputs':  input_data.tolist()}
data_json = json.dumps(data_obj)
#print(data_json)

start_time = time.time()
request_url = "http://localhost:8001/v1/models/unet:predict"
result = requests.post(request_url, data=data_json)
result_dict = json.loads(result.text)
delta = (time.time() - start_time)

print("Request URL:", request_url)
print("Input shape:", input_data.shape)
print("Output shape:", np.asarray(result_dict['outputs']).shape)
print("Completed Inference with one sample in {:.3f} sec,".format(delta))
