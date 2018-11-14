# AI_for_prosthetics
NIPS 2018 competition: AI for prothetics
Process of training and testing
1. Install
   - Install Anaconda3
   - Install packages from environment.yml
2. Training
   - cd opensim_http
   - ./run.sh
   - cd ../training
   - ipython -i ddpg2_v2.py
       + newpool(num_pools)
       + r(0,1000000)  # to run 1000000 episodes of training

   Note: the training models are saved in model2 folder

3. Testing:
   - cd opensim_http
   - python server -l 127.0.0.1 -p 5000
   open new command line
   - cd training
   - ipython -i ddpg2_test.py
       + agent.load_weights(<saved episode>) # load saved episode in models (please copy your selected episode weight to models folder)
       + test()
4. Challenge: follow the instructions in nips2018-ai-for-prosthetics-round2-starter-kit instructions

5. Reference
This work is inspired from https://github.com/ctmakro/stanford-osrl
   
