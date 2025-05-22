import librosa
import soundfile as sf

# Path to your ASVspoof audio file (choose a bonafide one)
flac_path = r"C:\Voice_Project\data\LA\ASVspoof2019_LA_train\flac\LA_T_1000137.flac"
output_path = "input.wav"

# Load and resample to match training settings
y, sr = librosa.load(flac_path, sr=16000)

# Save as input.wav to be used by your prediction script
sf.write(output_path, y, sr)

print("✅ Converted successfully to input.wav")
