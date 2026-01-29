import torch

# options
nb_objects = 5
quest_length = 2

num_cpu = 2
think_new_token = 32
action_new_token = 16
test_max_steps = 30 # test 30, GRPO 5
GRPO_max_steps = 5
model_name = "Qwen/Qwen3-1.7B"
test_txt_name = 'test_result.txt'
train_txt_name = 'train_result.txt'
train_data_pth = "./train_data/data/textworld_sft.jsonl"
sft_model_pth = "./model/save_model/qwen3_sft.pth"
grpo_model_pth = "./model/save_model/qwen3_grpo.pth"

inference_type = 'ReAct-Im' # 'ReAct', 'ReAct-Im', 'Act'

train_epoch_num = 1
grpo_updates = 1000
test_epoch_num = 5
test_epoch_steps = 1000

# get data hyperparam
get_data_num = 100000
train_seed_num = get_data_num // 2 + 1

device = "cuda" if torch.cuda.is_available() else "cpu"

lr = 2e-5
bs = 4

group_size = 4        # 같은 prompt에서 뽑을 rollout 수
kl_coef = 0.04