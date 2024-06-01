# British + American
test_spk = ["p225", "p230", "p270", "p274"]+["p294", "p300", "p329", "p330"]
train_spk = []
file = open("/mnt/iusers01/fatpou01/compsci01/n70579mp/scratch/datasets/speech/vctk/styletts2/emb/speaker_info.txt")
for line in file:
    items = line.strip().split()
    spk_id = items[0]
    if spk_id not in test_spk:
        train_spk.append(spk_id)
print(len(train_spk))
