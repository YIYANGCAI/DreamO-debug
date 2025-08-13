#!/bin/bash
#SBATCH --job-name=bagel-test          # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # number of tasks per node (adjust when using MPI)
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks, adjust when using OMP)
#SBATCH --time=24:00:00          # total run time limit (D-HH:MM:SS)
#SBATCH --gpus-per-node=1        # gpu
#SBATCH --partition=digital-human        # partition(queue) where you submit (amd/intel/gpu-a30/gpu-l20)
#SBATCH -o "./logs/${%j}-indexing.txt"

source activate dreamo

# python inference.py \
#     --ref1 /aifs4su/caiyiyang/new_method_play/personalization/DreamO/custom_test_examples/example1/woman1.png \
#     --ref2 /aifs4su/caiyiyang/new_method_play/personalization/DreamO/custom_test_examples/example1/shirt.png \
#     --task1 "id" \
#     --task2 "ip" \
#     --prompt "a girl wearing a shirt on a beach" \
#     --output "example1.png"

# python inference.py \
#     --ref1 /aifs4su/caiyiyang/new_method_play/personalization/DreamO/custom_test_examples/example2/hinton.jpeg \
#     --ref2 /aifs4su/caiyiyang/new_method_play/personalization/DreamO/custom_test_examples/example2/man1.png \
#     --task1 "id" \
#     --task2 "id" \
#     --prompt "a man is speaking with another man in a garden" \
#     --output "example2.png"

# python inference.py \
#     --ref1 /aifs4su/caiyiyang/new_method_play/personalization/DreamO/custom_test_examples/example3/dog2.png \
#     --ref2 /aifs4su/caiyiyang/new_method_play/personalization/DreamO/custom_test_examples/example3/toy1.png \
#     --task1 "ip" \
#     --task2 "ip" \
#     --prompt "a dog is playing with a toy on a table" \
#     --output "example3.png"

# python inference.py \
#     --ref1 /aifs4su/caiyiyang/new_method_play/personalization/DreamO/custom_test_examples/example4/1.jpg \
#     --ref2 /aifs4su/caiyiyang/new_method_play/personalization/DreamO/custom_test_examples/example4/2.jpg \
#     --task1 "id" \
#     --task2 "id" \
#     --prompt "an girl elf is talking with a handsome man in a office" \
#     --output "example4.png"

# python inference.py \
#     --ref1 /aifs4su/caiyiyang/new_method_play/personalization/DreamO/custom_test_examples/example5/nv.jpg \
#     --ref2 /aifs4su/caiyiyang/new_method_play/personalization/DreamO/custom_test_examples/example5/sazabi.jpg \
#     --task1 "ip" \
#     --task2 "ip" \
#     --prompt "a robot is fighting with another robot in the universe." \
#     --output "example5.png"

python inference.py \
    --ref1 /aifs4su/caiyiyang/new_method_play/personalization/DreamO/custom_test_examples/example6/kazuhiko.jpg \
    --ref2 /aifs4su/caiyiyang/new_method_play/personalization/DreamO/custom_test_examples/example6/chika.jpg \
    --task1 "ip" \
    --task2 "ip" \
    --prompt "a boy [ref#1] standing under a tree and a girl [ref#2] is sitting besides him" \
    --output "example6.png"

python inference.py \
    --ref1 /aifs4su/caiyiyang/new_method_play/personalization/DreamO/custom_test_examples/example7/kazuhiko.jpg \
    --ref2 /aifs4su/caiyiyang/new_method_play/personalization/DreamO/custom_test_examples/example7/anna.jpg \
    --task1 "ip" \
    --task2 "ip" \
    --prompt "a boy [ref#1] standing under a tree and a girl [ref#2] is sitting besides him" \
    --output "example7.png"

python inference.py \
    --ref1 /aifs4su/caiyiyang/new_method_play/personalization/DreamO/custom_test_examples/example8/anna.jpg \
    --ref2 /aifs4su/caiyiyang/new_method_play/personalization/DreamO/custom_test_examples/example8/chika.jpg \
    --task1 "ip" \
    --task2 "ip" \
    --prompt "a girl [ref#1] standing under a tree and a girl [ref#2] is sitting besides her" \
    --output "example8.png"

python inference.py \
    --ref1 /aifs4su/caiyiyang/new_method_play/personalization/DreamO/custom_test_examples/example8/chika.jpg \
    --ref2 /aifs4su/caiyiyang/new_method_play/personalization/DreamO/custom_test_examples/example8/anna.jpg \
    --task1 "ip" \
    --task2 "ip" \
    --prompt "a girl [ref#1] standing under a tree and a girl [ref#2] is sitting besides her" \
    --output "example9.png"