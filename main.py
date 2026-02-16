import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.video import r3d_18, R3D_18_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import os
from tqdm import tqdm
import glob
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pickle
import pandas as pd
from sklearn.metrics import roc_curve, auc

warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class VideoAudioDataset(Dataset):

    def __init__(self, root_dir, num_frames=20, audio_length=10, transform=None):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.audio_length = audio_length
        self.transform = transform
        self.sample_rate = 16000
        self.n_mels = 128
        self.label_map = {
            "RealVideo-RealAudio": (0, 0), "RealVideo-FakeAudio": (0, 1),
            "FakeVideo-RealAudio": (1, 0), "FakeVideo-FakeAudio": (1, 1)
        }
        self.samples = []
        video_extensions = ('*.mp4', '*.avi', '*.mov', '*.mkv')
        for label_folder, labels in self.label_map.items():
            label_path = os.path.join(root_dir, label_folder)
            if not os.path.isdir(label_path):
                print(f"Warning: Directory not found {label_path}")
                continue
            for ext in video_extensions:
                for video_path in glob.glob(os.path.join(label_path, '**', ext), recursive=True):
                    self.samples.append((video_path, labels))
        if not self.samples:
            raise ValueError(f"No video files found in {root_dir}. Please check the path and subdirectories.")
        print(f"Found {len(self.samples)} video files in dataset.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, (video_label, audio_label) = self.samples[idx]
        try:
            video, audio, info = torchvision.io.read_video(video_path, pts_unit='sec', end_pts=15)
            if video.size(0) == 0: return self._get_dummy_item() # Handle empty videos
        except Exception as e:
            return self._get_dummy_item()

        # Video processing
        if video.size(0) < self.num_frames:
            indices = torch.arange(video.size(0)).repeat(self.num_frames // video.size(0) + 1)[:self.num_frames]
        else:
            indices = torch.linspace(0, video.size(0) - 1, self.num_frames).long()

        frames = video[indices].permute(0, 3, 1, 2).float() / 255.0 # (T, C, H, W)

        if self.transform:
            frames = self.transform(frames)

        # Audio processing
        target_samples = self.sample_rate * self.audio_length
        if audio.numel() == 0: audio = torch.zeros(1, target_samples)
        else:
            if audio.dim() > 1 and audio.shape[0] > 1: audio = audio.mean(dim=0, keepdim=True)
            if info.get("audio_fps", self.sample_rate) != self.sample_rate:
                try:
                    resampler = torchaudio.transforms.Resample(orig_freq=info["audio_fps"], new_freq=self.sample_rate)
                    audio = resampler(audio)
                except Exception:
                    audio = torch.zeros(1, target_samples)
            if audio.shape[1] < target_samples: audio = F.pad(audio, (0, target_samples - audio.shape[1]))
            else:
                start = (audio.shape[1] - target_samples) // 2
                audio = audio[:, start:start + target_samples]

        mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate, n_fft=1024, hop_length=512, n_mels=self.n_mels)(audio)
        log_mel = torch.log(mel_spectrogram + 1e-9)
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-9)

        joint_label_map = {(0,0):0, (0,1):1, (1,0):2, (1,1):3}
        joint_label = joint_label_map.get((video_label, audio_label), -1)

        return {"video": frames, "audio": log_mel, "video_label": torch.tensor(video_label).float(),
                "audio_label": torch.tensor(audio_label).float(), "joint_label": torch.tensor(joint_label)}

    def _get_dummy_item(self):
        return {"video": torch.zeros((self.num_frames, 3, 224, 224)), "audio": torch.zeros((1, self.n_mels, 313)),
                "video_label": torch.tensor(0.0), "audio_label": torch.tensor(0.0), "joint_label": torch.tensor(-1)}


print("Setting up data augmentation and dataloaders...")
gpu_transform = nn.Sequential(
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
).to(device)

cpu_transform = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),
])

try:
    dataset = VideoAudioDataset(
        root_dir="/home/cvpr_phd_9/MMDFD/FakeAVCeleb_v1.2/",  # replace path
        num_frames=20, audio_length=10, transform=cpu_transform
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    if len(dataset) == 0: raise RuntimeError("Dataset is empty.")
except Exception as e:
    print(f"FATAL: Dataset creation failed: {e}")
    dataloader = []


class PretrainedUnifiedModel(nn.Module):
    def __init__(self, device='cuda'):
        super(PretrainedUnifiedModel, self).__init__()
        print("Initializing pre-trained models...")
        video_backbone = r3d_18(weights=R3D_18_Weights.DEFAULT)
        self.video_model = nn.Sequential(*list(video_backbone.children())[:-1])
        self.video_fc = nn.Linear(512, 512)
        audio_backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        audio_in_features = audio_backbone.classifier[1].in_features
        audio_backbone.classifier = nn.Linear(audio_in_features, 512)
        self.audio_model = audio_backbone
        self.fusion = nn.Sequential(
            nn.Linear(512 + 512, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3)
        )
        self.video_head = nn.Linear(512, 1); self.audio_head = nn.Linear(512, 1); self.joint_head = nn.Linear(512, 4)

    def forward(self, video_input, audio_input):
        video_features = self.video_model(video_input.permute(0, 2, 1, 3, 4))
        video_features = self.video_fc(video_features.flatten(1))
        audio_features = self.audio_model(audio_input.repeat(1, 3, 1, 1))
        fused = self.fusion(torch.cat((video_features, audio_features), dim=1))
        return {"video": self.video_head(fused), "audio": self.audio_head(fused), "joint": self.joint_head(fused)}

def calculate_class_weights(dataloader, device):
    joint_counts = np.zeros(4)
    for batch in dataloader:
        if -1 in batch["joint_label"]: continue
        for i in range(4): joint_counts[i] += (batch["joint_label"] == i).sum().item()
    total_samples = joint_counts.sum()
    if total_samples == 0: return torch.ones(4, dtype=torch.float32)
    weights = total_samples / (4 * (joint_counts + 1e-6))
    print(f"Calculated Class Weights: {weights}")
    return torch.tensor(weights, dtype=torch.float32).to(device)

def weighted_loss(outputs, targets, weights):

    video_loss = F.binary_cross_entropy_with_logits(outputs["video"], targets["video_label"])
    audio_loss = F.binary_cross_entropy_with_logits(outputs["audio"], targets["audio_label"])

    joint_loss = F.cross_entropy(
        outputs["joint"],
        targets["joint_label"].long(),
        weight=weights
    )

    return 0.25 * video_loss + 0.25 * audio_loss + 0.5 * joint_loss



def train_finetune(model, dataloader, optimizer, scheduler, weights, epochs=10, finetune_epoch=3):
    print("\n--- Starting Model Training ---")
    for name, param in model.named_parameters(): 
        if "video_model" in name or "audio_model.features" in name: param.requires_grad = False
    print("Stage 1: Training head layers only...")

    for epoch in range(epochs):
        if epoch == finetune_epoch:
            print(f"\n--- Stage 2: Unfreezing all layers for fine-tuning (Epoch {epoch+1}) ---\n")
            for param in model.parameters(): param.requires_grad = True

        model.train()
        total_loss, joint_correct, total_samples = 0.0, 0, 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            if -1 in batch["joint_label"]: continue
            video, audio, video_labels, audio_labels, joint_labels = (
                batch["video"].to(device), batch["audio"].to(device),
                batch["video_label"].to(device).view(-1, 1), batch["audio_label"].to(device).view(-1, 1),
                batch["joint_label"].to(device)
            )

            b, t, c, h, w = video.shape
            video = gpu_transform(video.view(b * t, c, h, w)).view(b, t, c, h, w)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(video, audio)

            loss = weighted_loss(outputs, {"video_label": video_labels, "audio_label": audio_labels, "joint_label": joint_labels}, weights)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_samples += joint_labels.size(0)
            joint_correct += (torch.argmax(outputs["joint"], dim=1) == joint_labels).sum().item()
            progress_bar.set_postfix({"Loss": f"{total_loss / (progress_bar.n + 1):.4f}", "Joint Acc": f"{joint_correct / total_samples:.4f}"})

        scheduler.step()
        print(f"Epoch {epoch+1} Summary: Loss: {total_loss / len(dataloader):.4f} | Joint Acc: {joint_correct / total_samples:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    print("\nFine-tuning complete!")
    return model

def evaluate_model_full(model, dataloader, device):
    print("\n--- Evaluating Model Performance ---")
    model.eval()
    true_j, pred_j, true_v, pred_v, true_a, pred_a = [], [], [], [], [], []
    eval_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to(device)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if -1 in batch["joint_label"]: continue
            video, audio = batch["video"].to(device), batch["audio"].to(device)
            b, t, c, h, w = video.shape
            video = eval_norm(video.view(b * t, c, h, w)).view(b, t, c, h, w)
            outputs = model(video, audio)
            
            true_j.extend(batch["joint_label"].numpy())
            true_v.extend(batch["video_label"].numpy())
            true_a.extend(batch["audio_label"].numpy())
            
            pred_j.extend(torch.argmax(outputs["joint"], dim=1).cpu().numpy())
            pred_v.extend((torch.sigmoid(outputs["video"]) > 0.5).int().cpu().numpy())
            pred_a.extend((torch.sigmoid(outputs["audio"]) > 0.5).int().cpu().numpy())


    if not true_j:
        print("Evaluation skipped: No valid samples found in the dataloader.")
        return


    labels = ["RV-RA", "RV-FA", "FV-RA", "FV-FA"]
    cm = confusion_matrix(true_j, pred_j, labels=list(range(4)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, annot_kws={"size": 14})
    plt.title("Joint Video-Audio Confusion Matrix", fontsize=16)
    plt.xlabel("Predicted Labels", fontsize=12)
    plt.ylabel("True Labels", fontsize=12)
    plt.tight_layout()
    

    cm_path = "/home/cvpr_phd_9/MMDFD/joint_confusion_matrix.png" # replace path
    plt.savefig(cm_path)
    plt.close() 
    print(f"\nConfusion matrix saved to {cm_path}")


    report_joint_dict = classification_report(true_j, pred_j, target_names=labels, digits=4, zero_division=0, output_dict=True)
    report_video_dict = classification_report(true_v, pred_v, target_names=["Real", "Fake"], digits=4, zero_division=0, output_dict=True)
    report_audio_dict = classification_report(true_a, pred_a, target_names=["Real", "Fake"], digits=4, zero_division=0, output_dict=True)

    df_joint = pd.DataFrame(report_joint_dict).transpose().round(4)
    df_video = pd.DataFrame(report_video_dict).transpose().round(4)
    df_audio = pd.DataFrame(report_audio_dict).transpose().round(4)

    report_text = "--- Classification Reports ---\n\n"
    report_text += "Joint Modality Report:\n" + df_joint.to_string() + "\n\n"
    report_text += "Video Modality Report:\n" + df_video.to_string() + "\n\n"
    report_text += "Audio Modality Report:\n" + df_audio.to_string()


    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis('off') # Hide the axes

    fig.text(0.05, 0.95, report_text, transform=ax.transAxes,
             fontfamily='monospace', size=12, verticalalignment='top')
    
    plt.tight_layout()
    
    report_path = "/home/cvpr_phd_9/MMDFD/classification_reports.png" # replace path
    plt.savefig(report_path, dpi=300) 
    plt.close()
    print(f"Classification reports saved to {report_path}")

    print("\n" + report_text)

def predict_deepfake(model, video_path):
 
    print(f"\n--- Running prediction on: {os.path.basename(video_path)} ---")
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    model.eval()

    try:
        num_frames = 20
        audio_length = 10
        sample_rate = 16000
        n_mels = 128

        cpu_transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
        ])
        eval_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to(device)


        video, audio, info = torchvision.io.read_video(video_path, pts_unit='sec', end_pts=15)
        if video.size(0) == 0:
            raise ValueError("Video stream is empty or could not be read.")

        if video.size(0) < num_frames:
            indices = torch.arange(video.size(0)).repeat(num_frames // video.size(0) + 1)[:num_frames]
        else:
            indices = torch.linspace(0, video.size(0) - 1, num_frames).long()

        frames = video[indices].permute(0, 3, 1, 2).float() / 255.0
        processed_video = cpu_transform(frames)
        processed_video = processed_video.unsqueeze(0).to(device)

        b, t, c, h, w = processed_video.shape
        processed_video = eval_norm(processed_video.view(b * t, c, h, w)).view(b, t, c, h, w)


        target_samples = sample_rate * audio_length
        if audio.numel() == 0: audio = torch.zeros(1, target_samples)
        else:
            if audio.dim() > 1: audio = audio.mean(dim=0, keepdim=True)
            if info.get("audio_fps", sample_rate) != sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=info["audio_fps"], new_freq=sample_rate)
                audio = resampler(audio)
            if audio.shape[1] < target_samples: audio = F.pad(audio, (0, target_samples - audio.shape[1]))
            else:
                start = (audio.shape[1] - target_samples) // 2
                audio = audio[:, start:start + target_samples]

        mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, hop_length=512, n_mels=n_mels)(audio)
        log_mel = torch.log(mel_spectrogram + 1e-9)
        processed_audio = ((log_mel - log_mel.mean()) / (log_mel.std() + 1e-9)).unsqueeze(0).to(device)


        with torch.no_grad():
            outputs = model(processed_video, processed_audio)


        video_fake_prob = torch.sigmoid(outputs["video"]).item()
        audio_fake_prob = torch.sigmoid(outputs["audio"]).item()
        joint_probs = torch.softmax(outputs["joint"], dim=1).squeeze().cpu().numpy()

        class_labels = ["RealVideo-RealAudio", "RealVideo-FakeAudio", "FakeVideo-RealAudio", "FakeVideo-FakeAudio"]
        joint_label = class_labels[np.argmax(joint_probs)]

        print(f"  - Primary Joint Prediction: {joint_label}")
        print(f"  - Video Head: {'FAKE' if video_fake_prob > 0.5 else 'REAL'} (Fake Probability: {video_fake_prob:.4f})")
        print(f"  - Audio Head: {'FAKE' if audio_fake_prob > 0.5 else 'REAL'} (Fake Probability: {audio_fake_prob:.4f})")

        print("  - Joint Prediction Probabilities:")
        for i, label in enumerate(class_labels):
            print(f"      {label}: {joint_probs[i]:.4f}")

    except Exception as e:
        print(f"An error occurred during prediction: {e}")

def evaluate_model_binary(model, dataloader, device):

    print("\n--- Performing Binary (Real vs. Fake) Evaluation ---")
    model.eval()
    true_binary_labels = []
    pred_binary_labels = []
    
    eval_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to(device)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Binary Evaluation"):
            if -1 in batch["joint_label"]: continue
            
            video, audio = batch["video"].to(device), batch["audio"].to(device)
            b, t, c, h, w = video.shape
            video = eval_norm(video.view(b * t, c, h, w)).view(b, t, c, h, w)
            outputs = model(video, audio)
            

            true_joint = batch["joint_label"].numpy()
            true_binary = (true_joint > 0).astype(int)
            true_binary_labels.extend(true_binary)
            
            pred_joint = torch.argmax(outputs["joint"], dim=1).cpu().numpy()
            pred_binary = (pred_joint > 0).astype(int)
            pred_binary_labels.extend(pred_binary)

    if not true_binary_labels:
        print("Binary evaluation skipped: No valid samples found.")
        return


    binary_labels = ["Real", "Fake"]
    report_dict = classification_report(
        true_binary_labels, 
        pred_binary_labels, 
        target_names=binary_labels, 
        digits=4, 
        zero_division=0, 
        output_dict=True
    )
    df_report = pd.DataFrame(report_dict).transpose().round(4)
    report_text = "--- Binary Classification Report (Real vs. Any Fake) ---\n\n" + df_report.to_string()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis('off')
    fig.text(0.05, 0.95, report_text, transform=ax.transAxes, fontfamily='monospace', size=12, verticalalignment='top')
    plt.tight_layout()
    
    report_path = "binary_classification_report.png"
    plt.savefig(report_path, dpi=300)
    plt.close()
    print(f"\nBinary classification report saved to {report_path}")
    print("\n" + report_text)


    cm = confusion_matrix(true_binary_labels, pred_binary_labels, labels=[0, 1])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=binary_labels, yticklabels=binary_labels, annot_kws={"size": 16})
    plt.title("Binary Confusion Matrix (Real vs. Fake)", fontsize=16)
    plt.xlabel("Predicted Labels", fontsize=12)
    plt.ylabel("True Labels", fontsize=12)
    plt.tight_layout()
    
    cm_path = "binary_confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    print(f"Binary confusion matrix saved to {cm_path}")

def plot_binary_auc_curve(model, dataloader, device):

    print("\n--- Generating Binary ROC AUC Curve ---")
    model.eval()
    true_binary_labels = []
    fake_probability_scores = []

    eval_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to(device)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating AUC Scores"):
            if -1 in batch["joint_label"]: continue

            video, audio = batch["video"].to(device), batch["audio"].to(device)
            b, t, c, h, w = video.shape
            video = eval_norm(video.view(b * t, c, h, w)).view(b, t, c, h, w)
            outputs = model(video, audio)

            true_joint = batch["joint_label"].numpy()
            true_binary = (true_joint > 0).astype(int)
            true_binary_labels.extend(true_binary)

            probs = torch.softmax(outputs["joint"], dim=1).cpu().numpy()


            fake_scores = 1 - probs[:, 0]
            fake_probability_scores.extend(fake_scores)

    if not true_binary_labels:
        print("AUC curve generation skipped: No valid samples found.")
        return

    fpr, tpr, thresholds = roc_curve(true_binary_labels, fake_probability_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='No-Skill Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True)
    plt.tight_layout()

    auc_path = "binary_roc_auc_curve.png"
    plt.savefig(auc_path)
    plt.close()
    print(f"ROC AUC curve saved to {auc_path}")

if __name__ == '__main__':

    if dataloader and len(dataloader) > 0:
        #----------------------------------------------------- TRAINING CODE -----------------------------------------------------#
        try:
            dataset = VideoAudioDataset(
                root_dir="/home/cvpr_phd_9/MMDFD/train_data/",  # replace path
                num_frames=20, audio_length=10, transform=cpu_transform
            )
            dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
            if len(dataset) == 0: raise RuntimeError("Dataset is empty.")
        except Exception as e:
            print(f"FATAL: Dataset creation failed: {e}")
            dataloader = []
        model = PretrainedUnifiedModel(device=device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
        NUM_EPOCHS = 10
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

        # # class_weights = calculate_class_weights(dataloader, device)
        class_weights = torch.tensor([10.77199998, 10.77199998, 0.55474302, 0.49709275], dtype=torch.float32).to(device) # i used this because these are the value of weights and calulating weights with the algorithm takes around 30 mins so this saves time

        trained_model = train_finetune(
            model, dataloader, optimizer, scheduler, class_weights,
            epochs=NUM_EPOCHS, finetune_epoch=0 #3
        )
        filename = '/home/cvpr_phd_9/MMDFD/fine_tune_fulldataset_80-20.sav' # replace path and add filename.sav
        pickle.dump(model, open(filename, 'wb'))
        trained_model = pickle.load(open(filename, 'rb'))
        print("Model saved")

        #----------------------------------------------------- EVALUATION CODE -----------------------------------------------------#
        try:
            dataset = VideoAudioDataset(
                root_dir="/home/cvpr_phd_9/MMDFD/test_data/",  # replace with path of test data set
                num_frames=20, audio_length=10, transform=cpu_transform
            )
            dataloader_test = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
            if len(dataset) == 0: raise RuntimeError("Dataset is empty.")
        except Exception as e:
            print(f"FATAL: Dataset creation failed: {e}")
            dataloade_test = []
        
        # evaluate_model_binary(loaded_model, test_dataloader, device)  -------------> does binary evaluation by converting dataset into real-fake classes (used to get AUC and compare performance with other models)
            

        # plot_binary_auc_curve(loaded_model, test_dataloader, device) ---------------> Plots the ROC curve by converting into binary
        
        evaluate_model_full(trained_model, dataloader_test, device)
    else:
        print("\nDataloader is empty. Skipping training and evaluation.")

    
    #----------------------------------------------------- STANDALONE TESTING -----------------------------------------------------#
    # Here pass the model path directly and the test dataset path and perform evaluation. To run this, make sure to comment everything in 'main' except this.

    # SAVED_MODEL_PATH = '/home/cvpr_phd_9/MMDFD/fine_tune_no_weights2.sav'
    # TEST_DATA_DIR = "/home/cvpr_phd_9/MMDFD/FakeAVCeleb_v1.2"

    # print(f"Loading model from: {SAVED_MODEL_PATH}")

    # try:
    #     with open(SAVED_MODEL_PATH, 'rb') as f:
    #         loaded_model = pickle.load(f)
    #     loaded_model.to(device)
    #     print("Model loaded successfully.")
    # except Exception as e:
    #     print(f"Failed to load the model. Error: {e}")
    #     exit()


    # try:
    #     cpu_transform = transforms.Compose([
    #         transforms.Resize((224, 224), antialias=True),
    #     ])
    #     test_dataset = VideoAudioDataset(
    #         root_dir=TEST_DATA_DIR,
    #         num_frames=20,
    #         audio_length=10,
    #         transform=cpu_transform
    #     )
    #     test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

    #     if len(test_dataset) == 0:
    #         print("Test dataset is empty. Cannot perform evaluation.")
    #     else:
    #         # --- Call the evaluation functions ---
            
    #         # Call the multi-class evaluation (optional)
    #         evaluate_model_full(loaded_model, test_dataloader, device)

    #         # Call the binary report/confusion matrix evaluation
    #         # evaluate_model_binary(loaded_model, test_dataloader, device)
            
    #         # --- ADD THE CALL TO THE NEW AUC FUNCTION HERE ---
    #         # plot_binary_auc_curve(loaded_model, test_dataloader, device)

    # except Exception as e:
    #     print(f"An error occurred during evaluation. Error: {e}")