
## Validate on LFW
https://github.com/davidsandberg/facenet/wiki/Validate-on-LFW

### Align the LFW dataset

python src/align/align_dataset_mtcnn.py /local/Work/datasets/lfw/raw/ /local/Work/datasets/lfw/lfw_20180422_160 --image_size 160 --margin 32 --random_order --gpu_memory_fraction 0.25

python src/validate_on_lfw.py /local/Work/datasets/lfw/lfw_20180422_160 /local/Work/models/facenet/20170512-110547


https://github.com/davidsandberg/facenet/wiki/Train-a-classifier-on-own-images

### Train a classifier on LFW

python src/classifier.py TRAIN ~/datasets/lfw/lfw_mtcnnpy_160/ ~/models/facenet/20170511-185253/20170511-185253.pb ~/models/facenet/lfw_classifier.pkl --batch_size 1000 --min_nrof_images_per_class 40 --nrof_train_images_per_class 35 --use_split_dataset

python src/classifier.py CLASSIFY ~/datasets/lfw/lfw_mtcnnpy_160/ ~/models/facenet/20170511-185253/20170511-185253.pb ~/models/facenet/lfw_classifier.pkl --batch_size 1000 --min_nrof_images_per_class 40 --nrof_train_images_per_class 35 --use_split_dataset
