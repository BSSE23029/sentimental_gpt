import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
DATA_FILE = "sentinel_augmented_data.csv"
MODEL_SAVE_PATH = "sentinel_model.pth"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-4
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print(f"🚀 Training on: {DEVICE}")

# --- 1. DATASET CLASS ---
class SentinelDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# --- 2. HYBRID MODEL ARCHITECTURE ---
class SentinelNet(nn.Module):
    def __init__(self, n_classes=2):
        super(SentinelNet, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, 
                            batch_first=True, bidirectional=True)
        self.out = nn.Linear(64 * 2, n_classes)
        self.drop = nn.Dropout(p=0.3)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = bert_output.last_hidden_state
        hidden = hidden.permute(0, 2, 1) # Reshape for CNN
        cnn_out = self.conv1(hidden)
        cnn_out = self.relu(cnn_out)
        cnn_out = self.pool(cnn_out)
        lstm_in = cnn_out.permute(0, 2, 1) # Reshape for LSTM
        _, (hn, cn) = self.lstm(lstm_in)
        final_state = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1)
        output = self.drop(final_state)
        return self.out(output)

# --- 3. TRAINING ENGINE ---
def train_epoch(model, data_loader, loss_fn, optimizer, device, n_examples):
    model.train()
    losses = []
    correct_predictions = 0

    for d in tqdm(data_loader, desc="Training"):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # FIX: Changed .double() to .float() for MPS compatibility
    return correct_predictions.float() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    # FIX: Changed .double() to .float() for MPS compatibility
    return correct_predictions.float() / n_examples, np.mean(losses)

# --- MAIN ---
def main():
    # 1. Load Data
    df = pd.read_csv(DATA_FILE)
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Train Size: {len(df_train)}, Val Size: {len(df_val)}")

    # 2. Tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # 3. Data Loaders
    train_dataset = SentinelDataset(df_train.text.to_numpy(), df_train.label.to_numpy(), tokenizer, MAX_LEN)
    val_dataset = SentinelDataset(df_val.text.to_numpy(), df_val.label.to_numpy(), tokenizer, MAX_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # 4. Initialize Model
    model = SentinelNet(n_classes=2)
    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss().to(DEVICE)

    # 5. Training Loop
    history = {'train_acc': [], 'val_acc': []}
    
    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        train_acc, train_loss = train_epoch(model, train_loader, loss_fn, optimizer, DEVICE, len(df_train))
        print(f'Train loss {train_loss:.4f} accuracy {train_acc:.4f}')

        val_acc, val_loss = eval_model(model, val_loader, loss_fn, DEVICE, len(df_val))
        print(f'Val   loss {val_loss:.4f} accuracy {val_acc:.4f}')
        
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

    # 6. Save
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n💾 Model saved to {MODEL_SAVE_PATH}")
    print("SentinelGPT is ready.")

if __name__ == "__main__":
    main()