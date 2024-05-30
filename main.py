import cv2
import numpy as np
import pandas as pd
import supervision as sv

from pathlib import Path
from typing import Dict, List, Union
from ultralytics import YOLO

from utils.misc import Pawn
from utils.team_colors import KitColorAnalyzer


class CoachAssistant:
    def __init__(self, model_path: Union[str, Path], kits_colors: np.array):

        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.teams_color_analyzer = KitColorAnalyzer(guessed_kit_colors=kits_colors)

        self.teams_lineup = {}

        self.players_dict: Dict[int, List[Pawn]] = {}
        self.ball_dict = {}
        self.aver_bbox_width = None

        # self.kits_colors = cv2.cvtColor(np.expand_dims(kits_colors.astype('float32')/ 255, 0), cv2.COLOR_RGB2Lab).squeeze()

        self.raw_frames = []  
        self.output_frames = []


        self.FPS = None
        self.W = None
        self.H = None
        self.batch_size = 64


    def read_video(self, input_path: Union[str, Path]) -> None:
        input_path = str(input_path)
        print(f"Reading video from '{input_path}'...")
        cap = cv2.VideoCapture(input_path)
        self.FPS = int(cap.get(cv2.CAP_PROP_FPS))
        self.W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("capture props:", self.FPS, self.W, self.H)
        assert cap.isOpened(), f"Can't read '{input_path}' file."
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.raw_frames.append(frame)
        cap.release()
        print(f"{len(self.raw_frames)} frames were captured!")
    
    def save_video(self, out_path: Union[str, Path], with_annotation=True) -> None:
        print("Saving resulting file...")
        source = self.output_frames if with_annotation else self.raw_frames
        print(f"No. of frames to save: [{len(source)}]")
        out_path = str(out_path)
        print(f"Saving video to '{out_path}'")
        cc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(out_path, cc, self.FPS, (self.W, self.H))
        for frame in source:
            out.write(frame)
        out.release()
        self.raw_frames.clear()
        self.output_frames.clear()

    def detect_objects(self) -> List:
        detections = []
        print("Start object detection...")
        for i in range(0, len(self.raw_frames), self.batch_size):
            frames_batch = self.raw_frames[i:i+self.batch_size]
            detections.extend(self.model.predict(frames_batch, conf=0.1))
        return detections

    def track_objects(self) -> None:
        print("Start object tracking...")
        detections = self.detect_objects()

        cropp_per_id = {}

        aver_bbox_width = []  # to unify marker size

        for fid, detection in enumerate(detections):
            index2label: Dict[int, str] = detection.names
            label2index: Dict[str, int] = {v:k for k,v in index2label.items()}

            players_classes = (label2index["goalkeeper"], label2index["player"])
            
            # convert to supervision detection format
            detection_sv = sv.Detections.from_ultralytics(detection)

            # track objects
            tracked_detections = self.tracker.update_with_detections(detection_sv)
            
            playesr_on_frame = []
            frame = self.raw_frames[fid]

            for tracked_obj in tracked_detections:  
                x1, y1, x2, y2 = map(int, tracked_obj[0])  # bbox in form xyxy
                if tracked_obj[3] in players_classes:
                    aver_bbox_width.append(x2-x1)
                    track_id = tracked_obj[4]
                    is_in = cropp_per_id.get(track_id)
                    if is_in is None:
                        y2_ = y1 + (y2-y1)//2
                        # take top halg of the image
                        cropped = frame[y1:y2_, x1:x2, :]
                        cropp_per_id[track_id] = cropped

                    p = Pawn(
                        track_id=tracked_obj[4],
                        class_id=tracked_obj[3],
                        label=index2label[tracked_obj[3]],
                        bbox=tracked_obj[0],
                    )
                    playesr_on_frame.append(p)
                    
            self.players_dict[fid] = playesr_on_frame

            for frame_detection in detection_sv:
                cls_id = frame_detection[3]
    
                if cls_id == label2index["ball"]:
                    bbox = frame_detection[0]
                    self.ball_dict[fid] = bbox
        self.aver_bbox_width = np.mean(aver_bbox_width)
        self.interpolate_ball_trajectory()
        self.teams_color_analyzer.fit_colors(cropp_per_id.values())

    def draw_annotations(self) -> None:
        print("Draw annotations...")
        team_color = self.teams_color_analyzer.kmeans.cluster_centers_[[0,2], :]
        for frame_id, raw_frame in enumerate(self.raw_frames):
            frame = raw_frame.copy()


            # draw annotations
            for player in self.players_dict[frame_id]:
                color = team_color[player.team_id].astype(int).tolist()

                frame = self.draw_marker(frame, player.bbox, color, player.track_id)

            xyxy = self.ball_dict[frame_id]
            frame = self.draw_triangle(frame, xyxy, (0, 255, 0))

            self.output_frames.append(frame)
    
    def draw_marker(self, frame, bbox, color, object_id=None)-> np.array:
        x1, y1, x2, y2 = map(int, bbox)
        cx, cy = (x1+x2)//2, (y1+y2)//2
        if self.aver_bbox_width:
            bw = int(self.aver_bbox_width)  # mb get aver size
        else:
            bw = int(x2 - x1)

        # TODO: recaclucalte angle using camera projection
        perspective_coef = y1 /(self.H // 2)
        r1_, r2_ = map(int, (bw*perspective_coef, .35*bw*perspective_coef))

        frame = cv2.ellipse(
            frame,
            center=(cx, int(y2)),
            axes=(r1_, r2_),
            angle=0.0,
            startAngle=-45,
            endAngle=245,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
            )
        # draw lable with player id
        if object_id:
            rec_width = 40
            rec_height = 30
            # x1_rect = cx - bw - rec_width//2
            # x2_rect = cx - bw + rec_width//2
            x1_rect = cx - r1_ - rec_width//2
            x2_rect = cx - r1_ + rec_width//2
            y1_rect = y2 - rec_height//2
            y2_rect = y2 + rec_height//2

            frame = cv2.rectangle(
                frame,
                (x1_rect, y1_rect),
                (x2_rect, y2_rect),
                color,
                cv2.FILLED,
            )

            text_x_pad = 10 if object_id < 99 else 2
            frame = cv2.putText(
                frame,
                str(object_id),
                (x1_rect+text_x_pad, y1_rect+20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
                )

        return frame

    def interpolate_ball_trajectory(self):
        print("Prerform ball interpolation...")
        ball_positions = pd.DataFrame.from_dict(self.ball_dict, orient='index').reindex(range(len(self.raw_frames)))
        ball_positions = ball_positions.interpolate()
        ball_positions = ball_positions.bfill()

        self.ball_dict = {idx: box for idx, box in enumerate(ball_positions.to_numpy())}

    def draw_triangle(self, frame, bbox, fill_color) -> np.array:
        x1, y1, x2, y2 = map(int, bbox)
        cx = (x1+x2)//2
        vertices = np.array([
            [cx, y1],
            [cx-5, y1-5],
            [cx+5, y1-5],
        ])
        frame = cv2.drawContours(frame, [vertices], 0, fill_color, cv2.FILLED)
        frame = cv2.drawContours(frame, [vertices], 0, (0, 0, 0), 1)
        return frame

    def assign_players_to_teams(self):
        for fid, players in self.players_dict.items():
            frame = self.raw_frames[fid]
            for p in players:
                team_id = self.teams_lineup.get(p.track_id)
                if team_id is None:
                    x1, y1, x2, y2 = map(int, p.bbox)
                    y2_ = y1 + (y2-y1)//2
                    cropped = frame[y1:y2_, x1:x2, :]
                    team_id = self.teams_color_analyzer.get_player_team_lbl(cropped)
                p.team_id = team_id
                self.teams_lineup[p.track_id] = team_id
                

if __name__ == "__main__":

    root = Path(".")
    model_path = root / "models/trained/best.pt"
    dataset = Path(r'D:\DATASETS\DFL-Bundesliga Data Shootout')
    clip_name = "08fd33_4.mp4"

    team_kits_colors = np.array([
        [182, 252, 164],  # team 1
        [ 80,  92,  66],  # goalkeeper 1
        [221, 232, 250],  # team 2
        [180, 143, 109],  # goalkeeper 2
    ])

    
    ca = CoachAssistant(model_path=model_path, kits_colors=team_kits_colors)
    ca.read_video(dataset / clip_name)
    ca.track_objects()
    ca.assign_players_to_teams()
    ca.draw_annotations()
    ca.save_video('./tmp/output.avi')
