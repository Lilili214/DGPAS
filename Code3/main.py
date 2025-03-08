import utils

#todo: Change the model name
model_name = 'vgg19'

#Model stage division
#vgg19
stages = [[0, 1], [2], [3], [4], [5], [6], [7], [8], [9], [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]]

#VGG19
device_mapping = [2, 1, 3, 8, 9, 7, 5, 6, 4, 0]


def adaptive_schedule(devices, stages, device_mapping,help_flag, batch_size=4, micro_batch_size=32):
    """Adaptive scheduling algorithm supporting micro batch splitting"""

    # Initialize data structure
    microbatch_split = [[micro_batch_size] * len(stages) for _ in range(batch_size)]  # Record the data segmentation ratio for each stage
    device_timelines = {d['id']: [] for d in devices}  # Equipment task timeline
    stage_times = []  # Benchmark time for each stage

    # Retrieve runtime data
    run_time = utils.get_runtime_data('models.xlsx', model_name)

    # Pre calculation stage benchmark time (full data)
    for stage_idx, stage in enumerate(stages):
        dev_type = devices[device_mapping[stage_idx]]['type']
        stage_time = sum(utils.get_one_step_time(run_time, dev_type, step) for step in stage)
        stage_times.append(stage_time)


    for mb_idx in range(batch_size):
        print(f"\n=== Processing Micro-batch {mb_idx} ({micro_batch_size} samples) ===")
        current_data = micro_batch_size  # Current amount of data to be processed

        for stage_idx in range(len(stages)):
            # split_ratio = microbatch_split[mb_idx][stage_idx]

            # Obtain the main processing device
            main_dev = devices[device_mapping[stage_idx]]
            main_time = stage_times[stage_idx] * micro_batch_size
            main_start = max(get_available_time(main_dev['id'], mb_idx, device_timelines),
                             device_timelines[devices[device_mapping[stage_idx - 1]]['id']][-1][-1] if stage_idx > 0 else 0)
            main_end = main_start + main_time
            print(f"main_start:{main_start} main_end:{main_end}")

            # ===== Dynamic assistance judgment =====
            if stage_idx < len(stages) - 1 :
                helper_dev = devices[device_mapping[stage_idx + 1]]

                # Computing device available time window
                main_available = get_available_time(main_dev['id'], mb_idx, device_timelines)
                helper_available = get_available_time(helper_dev['id'], mb_idx, device_timelines)

                # Assistance condition: The equipment behind is idle
                if main_end > helper_available and help_flag:
                    split = min((main_end - helper_available)/stage_times[stage_idx],micro_batch_size - batch_size/2)/2
                    print(f"main_end:{main_end} help_ava:{helper_available} stage time:{stage_times[stage_idx]} split:{split}")
                    # Perform dynamic segmentation
                    # new_split = int(micro_batch_size*0.7)
                    # transfer_data = (micro_batch_size - new_split)
                    transfer_data = int(split)
                    new_split = (micro_batch_size - transfer_data)

                    # Calculate the processing time for assistive devices
                    helper_time = sum(
                        utils.get_one_step_time(run_time, helper_dev['type'], step)
                        for step in stages[stage_idx]
                    ) * transfer_data

                    # Update segmentation records
                    microbatch_split[mb_idx][stage_idx] = new_split
                    microbatch_split[mb_idx][stage_idx + 1] += (micro_batch_size - new_split)

                    # Record assistance device tasks (including transmission delays)
                    transfer_time = 0
                    helper_start = max(helper_available,main_start)+transfer_time
                    helper_end = helper_start + helper_time + transfer_time
                    device_timelines[helper_dev['id']].append((mb_idx, stage_idx, helper_start, helper_end))

                    # Record main device tasks
                    main_end = main_available + (stage_times[stage_idx] * new_split) + transfer_time
                    # main_end = max(main_end,helper_end)

                    device_timelines[main_dev['id']].append((mb_idx, stage_idx, main_available, main_end))

                    print(f"Stage {stage_idx}:  device{main_dev['id']} → device{helper_dev['id']}")
                    print(f"  split: {new_split/micro_batch_size * 100}%/{(micro_batch_size-new_split)/micro_batch_size *100}%")
                    print(f"  transfer data: {transfer_data} samples (using {transfer_time} ms)")
                    print(f"  helping time: {helper_time}ms")
                    continue

            device_timelines[main_dev['id']].append((mb_idx, stage_idx, main_start, main_end))
            print(f"Stage {stage_idx}: device{main_dev['id']} compute {current_data} samples")

    # Calculate the final time indicator
    total_time = max([event[-1] for dev in device_timelines.values() for event in dev]) / 1000
    bubble_time = calculate_bubble_time(device_timelines, len(stages))
    return total_time, bubble_time

# The follow-up time will be announced gradually