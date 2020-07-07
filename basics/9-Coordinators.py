import tensorflow as tf
from PIL import Image

list_of_images = [r'D:\IT\SelfTut\PluralSight-Intro_to_Tensor_Flow\Data\images\berkay-gumustekin-ngqyo2AYYnE-unsplash.jpg',
                  r'D:\IT\SelfTut\PluralSight-Intro_to_Tensor_Flow\Data\images\david-lezcano-m-Doa-GTrUw-unsplash.jpg',
                  r'D:\IT\SelfTut\PluralSight-Intro_to_Tensor_Flow\Data\images\fredrik-ohlander-tGBRQw52Thw-unsplash.jpg',
                  r'D:\IT\SelfTut\PluralSight-Intro_to_Tensor_Flow\Data\images\jamie-street-0nk6XZp7_1E-unsplash.jpg',
                  r'D:\IT\SelfTut\PluralSight-Intro_to_Tensor_Flow\Data\images\ryan-walton-AbNO2iejoXA-unsplash.jpg']

queue_of_file_names = tf.train.string_input_producer(list_of_images)
image_reader = tf.WholeFileReader()
with tf.Session() as sess:
    cord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=cord)
    image_list = []
    for i in range(len(list_of_images)):
        _, image_file = image_reader.read(queue_of_file_names)
        image = tf.image.decode_jpeg(image_file)
        image = tf.image.resize_images(image,[224,224])
        image.set_shape((224,224,3))
        img_out = sess.run(image)
        print(img_out.shape)
        Image.fromarray(img_out.astype('uint8'),'RGB').show()
        image_list.append(tf.expand_dims(img_out,0))
        print(image_list[-1].shape)

    cord.request_stop()
    cord.join(threads)
    writer = tf.summary.FileWriter('board',graph=sess.graph)
    for i in range(len(image_list)):
        summary = tf.summary.image("Image-"+str(i),image_list[i])

print("complete")

