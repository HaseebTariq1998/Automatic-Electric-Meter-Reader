from tensorflow.keras.models import load_model
import pytesseract
import os

total_frames=0
frames_processed=0
#my_CNN_with_blank.h5 smart_meter_reading_1_9
# Get the absolute path to the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Specify the path to the SavedModel file relative to the current directory
saved_model_path = os.path.join(current_dir, '../DigitsClassifierTrained/digits_classifier_trained.h5')
# Load the model using the absolute path
cnn_model = load_model(saved_model_path)
#model2 = load_model('my_CNN_kw.h5')
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
list = []
results = []
results_meterno=[]
dic = {}
type = -1
string = ''
MAX_FEATURES = 600
GOOD_MATCH_PERCENT = 0.10
