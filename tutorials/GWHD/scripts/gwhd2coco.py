import os
import sys
import json
from PIL import Image

mini = True

def main(image_dpath, csv_fpath, output_fpath):
    
    imdict = []
    anndict = []
    catedict = [{'id': 0, 'name': 'spike', 'supercategory': 'spike'}]
    
    with open(csv_fpath, 'r') as infh:
        for imgid, record in enumerate(infh):
            if record[0:10] == 'image_name':
                continue
            
            imgname, bboxes, tag = record.split(',')
            
            # check images in train subset or not
            if not os.path.exists(os.path.join(image_dpath, imgname + '.png')):
                continue
            
            w, h = Image.open(os.path.join(image_dpath, imgname + '.png')).size
            imdict.append({
                'id': imgid,
                'width': w,
                'height': h,
                'file_name':  '{}.png'.format(imgname),
            })
            
            if bboxes != 'no_box':
                for boxid, bbox in enumerate(bboxes.split(';')):
                    x1, y1, x2, y2 = [int(i) for i in bbox.split(' ')]
                    anndict.append({
                        'id': int('{}{:05}'.format(imgid, boxid)),
                        'image_id': imgid,
                        'category_id': 0,
                        'bbox': [x1, y1, x2 - x1 + 1, y2 - y1 + 1],
                        'area': (x2 - x1 + 1) * (y2 - y1 + 1) 
                    })
            
            #if mini and imgid > 9:
            #    break
    
    cocodict = {
        'images': imdict,
        'annotations': anndict,
        'categories': catedict,
    }
    
    with open(output_fpath, 'w') as infh:
        json.dump(cocodict, infh, indent=4)
    
    

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
    
    



