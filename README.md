# TensorFlow Lite image processing platform for Home Assistant

[![hacs_badge](https://img.shields.io/badge/HACS-Custom-orange.svg)](https://github.com/custom-components/hacs)
[![Donate](https://img.shields.io/badge/donate-Yandex-red.svg)](https://money.yandex.ru/to/4100110221014297)

The `tensorflow_lite` image processing platform allows you to detect and recognize objects in a camera image using [TensorFlow Lite](https://www.tensorflow.org/lite). The state of the entity is the number of objects detected, and recognized objects are listed in the `summary` attribute along with quantity. The `matches` attribute provides the confidence `score` for recognition and the bounding `box` of the object for each detection category.

Based on [Home Assistant tensorflow component](https://github.com/home-assistant/core/tree/dev/homeassistant/components/tensorflow) and [TensorFlow Lite Python object detection example](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/raspberry_pi).

## Installation instructions

1. Copy the contents of `custom_components/tensorflow_lite/` to `<your config dir>/custom_components/tensorflow_lite/` or use [HACS](https://hacs.xyz/docs/faq/custom_repositories/).

2. Install TensorFlow Lite Runtime package with pip (see the [official install guide](https://www.tensorflow.org/lite/guide/python) for other options):

```bash
pip3 install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime==2.5.0
```

Or installation in the `deps` directory:

```bash
pip3 install --no-dependencies --target /config/deps/lib/python3.8/site-packages --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime==2.5.0
```

>**Attention! There is currently no way to use the `tflite_runtime` package on [Home Assistant OS](https://github.com/home-assistant/operating-system). If you try to install the package in the `deps` directory, you will lose the ability to run your Home Assistant instance.**
>
>To fix this problem, we need to [build the wheel](https://www.tensorflow.org/lite/guide/build_cmake_pip) from the source code using `Alpine Linux` and `musl libc`. If you can do this, then please contact me.
>
>Instead, you can use the [DOODS integration](https://www.home-assistant.io/integrations/doods/) (but it needs more resources to do this).

3. Restart Home Assistant

## Configuration

To enable this platform in your installation, add the following to your `configuration.yaml` file:

```yaml
image_processing:
  - platform: tensorflow_lite
    source:
      - entity_id: camera.yourcamera
```

Then you need to restart Home Assistant again.

---

### CONFIGURATION VARIABLES

* **source** *(map) (Required)*  
  The list of image sources.

  * **entity_id** *(string) (Required)*  
    A camera entity id to get picture from.

  * **name** *(string) (Optional)*  
    This parameter allows you to override the name of your `image_processing` entity.

* **confidence** *(float) (Optional)*  
  The default confidence for any detected objects where not explicitly set.

* **model** *(map) (Optional)*  
  Information about the TensorFlow Lite model.

  * **file** *(string) (Optional)*  
    Full path to the base model file. For more information, see [official hosted models guide](https://www.tensorflow.org/lite/guide/hosted_models).  
    *Default:* /config/custom_components/tensorflow_lite/model.tflite ([COCO SSD MobileNet v1](https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/1?lite-format=tflite))

  * **labels** *(string) (Optional)*  
    Full path to a *label_map.pbtext.  
    *Default:* /config/custom_components/tensorflow_lite/[labels.txt](https://dl.google.com/coral/canned_models/coco_labels.txt)

  * **label_offset** *(integer) (Optional)*  
    Offset for mapping label ID to a name (only use for custom models).  
    *Default:* 0

  * **area** *(map) (Optional)*  
    Custom detection area. Only objects fully in this box will be reported. Top of image is 0, bottom is 1. Same left to right.

    * **top** *(float) (Optional)*  
      Top line defined as % from top of image.  
      *Default:* 0

    * **left** *(float) (Optional)*  
      Left line defined as % from left of image.  
      *Default:* 0

     * **bottom** *(float) (Optional)*  
       Bottom line defined as % from top of image.  
       *Default:* 1

     * **right** *(float) (Optional)*  
       Right line defined as % from left of image.  
       *Default:* 1

  * **categories** *(list) (Optional)*  
    List of categories to include in object detection. Can be seen in the file provided to `labels`.

* **file_out** *(list) (Optional)*  
  A [template](https://www.home-assistant.io/docs/configuration/templating/#processing-incoming-data) for the integration to save processed images including bounding boxes. `camera_entity` is available as the `entity_id` string of the triggered source camera.

---

`categories` can also be defined as dictionary providing an `area` for each category as seen in the advanced configuration below:

```yaml
# Example advanced configuration.yaml entry
image_processing:
  - platform: tensorflow_lite
    source:
      - entity_id: camera.driveway
      - entity_id: camera.backyard
    file_out:
      - "/tmp/{{ camera_entity.split('.')[1] }}_latest.jpg"
      - "/tmp/{{ camera_entity.split('.')[1] }}_{{ now().strftime('%Y%m%d_%H%M%S') }}.jpg"
    model:
      file: /config/ssd_mobilenet_v1_1_metadata_1.tflite
      categories:
        - category: person
          area:
            # Exclude top 10% of image
            top: 0.1
            # Exclude right 15% of image
            right: 0.85
        - car
        - truck
```

## Optimizing resources

[Image processing components](https://www.home-assistant.io/integrations/image_processing/) process the image from a camera at a fixed period given by the `scan_interval`. This leads to excessive processing if the image on the camera hasn't changed, as the default `scan_interval` is 10 seconds. You can override this by adding to your configuration `scan_interval: 10000` (setting the interval to 10,000 seconds), and then call the `image_processing.scan` service when you actually want to perform processing.

```yaml
# Example advanced configuration.yaml entry
image_processing:
  - platform: tensorflow_lite
    scan_interval: 10000
    source:
      - entity_id: camera.driveway
      - entity_id: camera.backyard
```

```yaml
# Example advanced automations.yaml entry
- alias: "TensorFlow Lite scanning"
  trigger:
     - platform: state
       entity_id:
         - binary_sensor.driveway
  action:
    - service: image_processing.scan
      target:
        entity_id: camera.driveway
```