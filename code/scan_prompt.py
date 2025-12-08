import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
import warnings

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
MODEL_PATH = "sentinel_model.pth"
MAX_LEN = 128
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- 1. RE-DEFINE ARCHITECTURE (Must match training exactly) ---
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
        hidden = bert_output.last_hidden_state.permute(0, 2, 1)
        cnn_out = self.pool(self.relu(self.conv1(hidden)))
        lstm_in = cnn_out.permute(0, 2, 1)
        _, (hn, cn) = self.lstm(lstm_in)
        final_state = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1)
        return self.out(self.drop(final_state))

# --- 2. LOAD MODEL ---
print("⏳ Loading SentinelGPT...")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = SentinelNet(n_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print("✅ SentinelGPT Active.\n")

# --- 3. PREDICTION FUNCTION ---
def scan_text(text):
    encoded = tokenizer.encode_plus(
        text,
        max_length=MAX_LEN,
        add_special_tokens=True,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoded['input_ids'].to(DEVICE)
    attention_mask = encoded['attention_mask'].to(DEVICE)

    with torch.no_grad():
        output = model(input_ids, attention_mask)
        probs = torch.nn.functional.softmax(output, dim=1)
        
    confidence, prediction = torch.max(probs, dim=1)
    return prediction.item(), confidence.item()

# --- 4. INTERACTIVE LOOP ---
print("--- 🛡️ SENTINEL PROMPT SCANNER 🛡️ ---")
print("Type a prompt to scan it. Type 'exit' to quit.\n")

while True:
    user_input = input(">> Enter Prompt: ")
    if user_input.lower() in ['exit', 'quit']:
        break
        
    label, conf = scan_text(user_input)
    
    if label == 1:
        print(f"🚨 ALERT: MALICIOUS PROMPT DETECTED! (Confidence: {conf*100:.2f}%)")
    else:
        print(f"✅ CLEAN: Prompt is Safe. (Confidence: {conf*100:.2f}%)")
    print("-" * 40)