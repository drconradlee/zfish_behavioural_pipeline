import numpy as np
import ffmpeg
import os

def run_segmentation(file_list,file_num,all_coord,num_wells):
    for well in np.arange(num_wells[file_num][0]):
        print(f'File Progress: {file_num+1}  (Currently Working on Well: {well+1}) ...')
        prog_length = 100;
        prog_percent = (well+1)/(num_wells[file_num][0])
        prog_comp = np.round(prog_percent*prog_length)

        print('['+'#'*int(prog_comp) + '-'*(prog_length-int(prog_comp)) +']    ' + str(int(prog_comp)) + '%')
        try:
            stream = ffmpeg.input(file_list[file_num])
            stream = ffmpeg.filter(stream,'crop',\
                                   str(round(all_coord[file_num][well][0])),\
                                   str(round(all_coord[file_num][well][1])),\
                                   str(round(all_coord[file_num][well][2])),\
                                   str(round(all_coord[file_num][well][3])),\
                                  )
            
            target_dir = os.path.join(os.path.dirname(file_list[file_num]), 'well_videos')
            base_name = os.path.splitext(os.path.basename(file_list[file_num]))[0]
            target_file = os.path.join(target_dir,base_name + '_well_' + str(well+1) + '.avi')

            stream = ffmpeg.output(stream, target_file, **{'c:v': 'libx264rgb'},\
                                  minrate='4000k', maxrate='4000k', preset='medium', crf=12, f='avi')
            stream = ffmpeg.overwrite_output(stream)
            ffmpeg.run(stream,capture_stdout=True, capture_stderr=True)
        except ffmpeg.Error as e:
            print('stdout:', e.stdout.decode('utf8'))
            print('stderr:', e.stderr.decode('utf8'))
            raise e