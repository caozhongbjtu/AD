import os

def create_log_file(log,time_, cs_file):
    os.makedirs(log, exist_ok=True)
    os.makedirs(log+"_model", exist_ok=True)
    filename = f"time_{time_}_{cs_file}.txt"
    model_name =  f"time_{time_}_{cs_file}"
    file_path = os.path.join(log, filename)
    model_path = os.path.join(log+"_model", model_name)
    # model_path = os.path.join(log, model_name)
    f = open(file_path, "w")
    return f, file_path,model_path

def log_print(f, text):
    print(text)
    f.write(text + "\n")
    f.flush()

def write_config(f, args):
    f.write("===== Config =====\n")
    for k, v in vars(args).items():
        f.write(f"{k} = {v}\n")
    f.write("\n")