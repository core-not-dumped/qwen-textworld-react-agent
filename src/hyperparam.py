# options
nb_objects = 5
quest_length = 2

num_cpu = 1
think_new_token = 32
action_new_token = 16
max_steps = 30
model_name = "Qwen/Qwen3-1.7B"
test_txt_name = 'result.txt'

inference_type = 'Act' # 'ReAct', 'ReAct-Im', 'Act'

epoch_num = 5
epoch_steps = 1000

# get data hyperparam
get_data_num = 100000
train_seed_num = get_data_num // 2 + 1