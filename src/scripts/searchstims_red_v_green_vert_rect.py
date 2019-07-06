import copy
from pathlib import Path

import searchstims.make
from searchstims.stim_makers import RVvGVStimMaker, RVvRHGVStimMaker


COLOR_DIFFS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
MAX_RED_GREEN = [255, 255, 0]
REDS = []
GREENS = []
for diff in COLOR_DIFFS:
    diff_from_255 = round(255 - (255 * diff))

    a_red = copy.deepcopy(MAX_RED_GREEN)
    a_red[1] = diff_from_255
    a_red = tuple(a_red)
    REDS.append(a_red)

    a_green = copy.deepcopy(MAX_RED_GREEN)
    a_green[0] = diff_from_255
    a_green = tuple(a_green)
    GREENS.append(a_green)

ALEXNET_SIZE = (227, 227)
BORDER_SIZE = (30, 30)
GRID_SIZE = None
MIN_CENTER_DIST = 30
ITEM_BBOX_SIZE = (20, 20)
JITTER = 0

stim_dict = {}
for diff, target_color, distractor_color in zip(COLOR_DIFFS, REDS, GREENS):
    stim_name = f'red_v_green_rect_diff{diff}'
    stim_maker = RVvRHGVStimMaker(target_color=target_color,
                                  distractor_color=distractor_color,
                                  window_size=ALEXNET_SIZE,
                                  border_size=BORDER_SIZE,
                                  grid_size=GRID_SIZE,
                                  min_center_dist=MIN_CENTER_DIST,
                                  item_bbox_size=ITEM_BBOX_SIZE,
                                  jitter=JITTER)
    stim_dict[stim_name] = stim_maker

OUTPUT_DIR = Path('data/visual_search_stimuli/red_v_green_vert_rect')
JSON_FILENAME = 'red_v_green_vert_rect.json'
TARGET_PRESENT = 4800 // len(COLOR_DIFFS)
TARGET_ABSENT = 4800 // len(COLOR_DIFFS)
SET_SIZES = [1, 2, 4, 8]


def main():
    searchstims.make.make(root_output_dir=OUTPUT_DIR,
                          stim_dict=stim_dict,
                          json_filename=JSON_FILENAME,
                          num_target_present=TARGET_PRESENT,
                          num_target_absent=TARGET_ABSENT,
                          set_sizes=SET_SIZES)


if __name__ == '__main__':
    main()
