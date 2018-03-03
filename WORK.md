
## Validate on LFW
https://github.com/davidsandberg/facenet/wiki/Validate-on-LFW

### Align the LFW dataset

python3 src/align/align_dataset_mtcnn.py ~/datasets/lfw/raw/ ~/datasets/lfw/lfw_mtcnnpy_160 --image_size 160 --margin 32 --random_order --gpu_memory_fraction 0.25

python3 src/validate_on_lfw.py ~/datasets/lfw/lfw_mtcnnpy_160 ~/models/facenet/20170512-11054


https://github.com/davidsandberg/facenet/wiki/Train-a-classifier-on-own-images

### Train a classifier on LFW

python3 src/classifier.py TRAIN ~/datasets/lfw/lfw_mtcnnpy_160/ ~/models/facenet/20170511-185253/20170511-185253.pb ~/models/facenet/lfw_classifier.pkl --batch_size 1000 --min_nrof_images_per_class 40 --nrof_train_images_per_class 35 --use_split_dataset

python3 src/classifier.py CLASSIFY ~/datasets/lfw/lfw_mtcnnpy_160/ ~/models/facenet/20170511-185253/20170511-185253.pb ~/models/facenet/lfw_classifier.pkl --batch_size 1000 --min_nrof_images_per_class 40 --nrof_train_images_per_class 35 --use_split_dataset
