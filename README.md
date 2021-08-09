usage: texture_transfer.py [-h] --text TEXTURE_PATH --picture PIC_PATH
                           --outname OUTNAME [--overlap_len OVERLAP_LEN]
                           [--patchsize PATCHSIZE] [--random RANDOM]

optional arguments:
  -h, --help            show this help message and exit
 
 --text TEXTURE_PATH, -t TEXTURE_PATH
                        path of texture image

  --picture PIC_PATH, -p PIC_PATH
                        path of source image

  --outname OUTNAME, -o OUTNAME
                        name of the output image

  --overlap_len OVERLAP_LEN
                        the size of the overlap region between the patches

  --patchsize PATCHSIZE
                        size of the patches

  --random RANDOM       random choose patch in some similar patches, 0~9. This
                        value becomes bigger, then #candicates of patches
                        becomes larger. 0 is choose the most similar patch.
                        That is #candicates of patches is one.



---------------------------------------------------------------------------------

texture_transfer.py is the original algorithm.


texture_transfer_M.py is Modification. That is, only paste the texture's luminance and keep the original image's color.
