import events as e
RANDOM_SEED = None
BATCH_SIZE = 1024
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.1
EPS_DECAY = 10000
EPS_PLAY = 0.0
EPS_ALWAYS = 0.0
TARGET_UPDATE = 2000
TRAIN_UPDATE = 1
LOOPS_SIZE = 20
MEMORY_SIZE = 1000000
LEARNING_RATE = 0.0001
FRAMES = 1
FEATURE_SIZE = 21
TEMP = 100
TEMP_START = 100
TEMP_END = 5
TEMP_DECAY = 10000
MODE = "EPSILON"
CLAMPED = False
MODEL_PATH = "models/crate_destroyer.pt"
COIN_PATH = "models/coin_collector.pt"
VERBOSE = False
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']