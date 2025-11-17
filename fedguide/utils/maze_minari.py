import minari
ds = minari.load_dataset("D4RL/pointmaze/medium-v2", download=True)
print(len(ds))