# AudioKeyClassification

Even music professionals are not able to detect the audio keys in real world.
I built state of the art method of key classification(AllConvNet) and compared with shallow ResNet model.

# Files

5320 audio files (5 seconds long)
Preprocessed with signal_process.py as logmelspectrogram


## Model

Referred Genre-Agnostic_Key_Classification_With_Convolution reasearch paper for AllConvNet.
Applied ResNet for better results.

## Results

AllConvNet: 83% validation accuracy
ResNet: 85% validation accuracy
