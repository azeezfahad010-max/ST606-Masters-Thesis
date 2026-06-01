Football Video Analysis Using ComputerVision
Automatically detects players in a football video, tracks them across frames, and labels each one with their team using jersey colour.

What It Does:

- Detects players, the ball, and referees using YOLO
- Tracks each player and gives them a persistent ID
- Figures out which team each player belongs to by their jersey colour
- Outputs an annotated video with coloured boxes and labels


Files:

1. player_tracking_id.py: Tracks players and assigns stable IDs.
2. teamewise_classifier.py: Reads jersey colours and identifies the two teams.
3. main.py: Runs the full pipeline and produces the output video.
4. requirements.txt: Lists all the packages you need to install.

Setup:

- You will need: Python 3.10+, and a YOLO .pt weights file trained on football footage.
- Download trained_YOLO.pt and sample.mp4 from [the Google Drive folder ](https://drive.google.com/drive/folders/1QOsNy0KLvAHMYuvnS3R-d1VorXe4PnHW?usp=share_link) and place both files in the project folder which contains all the codes.
  
- Install dependencies:

bashpip install -r requirements.txt

- Running the Code:

1.Full pipeline (produces annotated video):

python main.py --model trained_YOLO.pt --video sample.mp4 --output football_out.mp4

2.Tracker only (prints player IDs per frame, no team labels):

python player_tracking_id.py --model trained_YOLO.pt --video sample.mp4 --frames 300

3.Classifier only (shows the two team colours it found):

python teamwise_classifier.py --model trained_YOLO.pt --video sample.mp4

Reading the Output Video:

Box colourMeaning:

- Red: Team 1 player
- Blue: Team 2 player
- Grey: Player detected but team not confirmed

Dependencies

- Ultralytics — YOLO detection
- Supervision — tracking utilities
- OpenCV — video processing
- scikit-learn — team colour clustering
- PyTorch — model runtime
