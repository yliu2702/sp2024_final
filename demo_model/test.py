from base.constant import *
from base.helper import *
import pandas as pd
import tqdm
from transformers import AutoTokenizer,AutoModelForSequenceClassification

CACHE_DIR = os.environ.get("TRANSFORMERS_CACHE")
BEST_MODEL_PATH = f"{DEMO_SAVE_PATH}/best_model"

## TODO: adjust hyperparameters
MAX_LEN = 256
TRAIN_BATCH_SIZE = 16 # 8? 32? 6?
VALID_BATCH_SIZE = 16
LEARNING_RATE = 1e-05

model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=6, cache_dir=CACHE_DIR)
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

test_df = pd.read_csv(DATA_BASE_DIR + "/model/test.csv")
test_data = CustomDataset(test_df, tokenizer, MAX_LEN)
test_data_loader = torch.utils.data.DataLoader(
    test_data,
    shuffle = True,
    batch_size = TRAIN_BATCH_SIZE,
    num_workers = 0
)

logger = Logger(f"{DEMO_SAVE_PATH}/test_logs.log")
test_loss = 0
test_acc = 0
for index, batch in tqdm(test_data_loader, total = len(test_data_loader)):
    input_ids = batch['input_ids'].to(device, dtype = torch.long)
    attention_mask = batch['attention_mask'].to(device, dtype = torch.long)
    token_type_ids = batch['token_type_ids'].to(device, dtype = torch.long)
    targets = batch['target'].to(device, dtype = torch.long) 
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, token_type_ids)
        loss = loss_fn(outputs.logits, targets)
        test_loss += loss.item()
        outputs_numpy = outputs.logits.detach().cpu().numpy()
        label_numpy = targets.to('cpu').numpy()
        test_acc += flat_acc(outputs_numpy, label_numpy)

avg_test_loss = test_loss/len(test_data_loader)
avg_test_acc = test_acc/len(test_data_loader)
logger.info("Test Loss: %.4f, Test Accuracy: %.4f", avg_test_loss, avg_test_acc)