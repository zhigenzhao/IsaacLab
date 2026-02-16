# Standard evaluation (no critic guidance)
# python scripts/tools/eval_lerobot_policy_g1.py --task Isaac-PickPlace-G1-InspireFTP-XR-v0 --policy_path kelvinzhaozg/vqvfm_g1_pick_place_0202 --record_video --enable_cameras --headless --video_dir ./eval_vqvfm_g1_pick_place_0202/video --num_episodes 10 --log_file ./eval_vqvfm_g1_pick_place_0202/episode_results.json --plugin_packages lerobot_policy_vqvfm

# Guided inference with critic (uncomment and set CRITIC_PATH):
python scripts/tools/eval_lerobot_policy_g1.py --task Isaac-PickPlace-G1-InspireFTP-XR-v0 --policy_path kelvinzhaozg/vqvfm_g1_pick_place_0202 --record_video --enable_cameras --headless --video_dir ./eval_vqvfm_g1_pick_place_0202_guided/video --num_episodes 10 --log_file ./eval_vqvfm_g1_pick_place_0202_guided/episode_results.json --plugin_packages lerobot_policy_vqvfm --policy_overrides critic_repo_path=kelvinzhaozg/vqvfm_critic_g1_pick_place_0202 guidance_weight_max=1.0
