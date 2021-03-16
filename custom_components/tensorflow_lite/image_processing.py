"""Support for performing TensorFlow Lite classification on images."""
import io
import logging
import os
import re
import time

from PIL import Image, ImageDraw, UnidentifiedImageError
import numpy as np
import tflite_runtime.interpreter as tflite
import voluptuous as vol

from homeassistant.components.image_processing import (
    PLATFORM_SCHEMA,
    CONF_SOURCE,
    CONF_ENTITY_ID,
    CONF_NAME,
    CONF_CONFIDENCE,
    ImageProcessingEntity
)
from homeassistant.const import EVENT_HOMEASSISTANT_START
from homeassistant.core import split_entity_id
from homeassistant.helpers import template
import homeassistant.helpers.config_validation as cv
from homeassistant.util.pil import draw_box

DOMAIN = "tensorflow_lite"
_LOGGER = logging.getLogger(__name__)

ATTR_MATCHES = "matches"
ATTR_SUMMARY = "summary"
ATTR_TOTAL_MATCHES = "total_matches"
ATTR_PROCESS_TIME = "process_time"

CONF_MODEL = "model"
CONF_FILE = "file"
CONF_LABELS = "labels"
CONF_LABEL_OFFSET = "label_offset"
CONF_AREA = "area"
CONF_TOP = "top"
CONF_LEFT = "left"
CONF_BOTTOM = "bottom"
CONF_RIGHT = "right"
CONF_CATEGORIES = "categories"
CONF_CATEGORY = "category"
CONF_FILE_OUT = "file_out"

AREA_SCHEMA = vol.Schema(
    {
        vol.Optional(CONF_BOTTOM, default=1): cv.small_float,
        vol.Optional(CONF_LEFT, default=0): cv.small_float,
        vol.Optional(CONF_RIGHT, default=1): cv.small_float,
        vol.Optional(CONF_TOP, default=0): cv.small_float,
    }
)

CATEGORY_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_CATEGORY): cv.string, vol.Optional(CONF_AREA): AREA_SCHEMA
    }
)

PLATFORM_SCHEMA = PLATFORM_SCHEMA.extend(
    {
        vol.Optional(CONF_MODEL, default={}): vol.Schema(
            {
                vol.Optional(CONF_FILE): cv.isfile,
                vol.Optional(CONF_LABELS): cv.isfile,
                vol.Optional(CONF_LABEL_OFFSET, default=0): int,
                vol.Optional(CONF_AREA): AREA_SCHEMA,
                vol.Optional(CONF_CATEGORIES, default=[]): vol.All(
                    cv.ensure_list, [vol.Any(cv.string, CATEGORY_SCHEMA)]
                )
            }
        ),
        vol.Optional(CONF_FILE_OUT, default=[]): vol.All(cv.ensure_list, [cv.template])
    }
)

def load_labels(path):
    """Loads the labels file. Supports files with or without index numbers."""
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        labels = {}
        for row_number, content in enumerate(lines):
            pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
            if len(pair) == 2 and pair[0].strip().isdigit():
                labels[int(pair[0])] = pair[1].strip()
            else:
                labels[row_number] = pair[0].strip()
    return labels

def setup_platform(hass, config, add_entities, discovery_info=None):
    """Set up the TensorFlow Lite image processing platform."""
    model_config = config[CONF_MODEL]
    model_file = model_config.get(CONF_FILE) or hass.config.path("custom_components", "tensorflow_lite", "model.tflite")
    labels =  model_config.get(CONF_LABELS) or hass.config.path("custom_components", "tensorflow_lite", "labels.txt")

    # Make sure locations exist
    if (
        not os.path.exists(model_file)
        or not os.path.exists(labels)
    ):
        _LOGGER.error("Unable to locate tensorflow lite model or label map")
        return

    try:
        # Display warning that PIL will be used if no OpenCV is found.
        import cv2  # noqa: F401 pylint: disable=unused-import, import-outside-toplevel
    except ImportError:
        _LOGGER.warning(
            "No OpenCV library found. TensorFlow Lite will process image with "
            "PIL at reduced resolution"
        )

    hass.data[DOMAIN] = {CONF_MODEL: None}

    def tensorflow_lite_hass_start(_event):
        """Set up TensorFlow Lite model on hass start."""

        model = tflite.Interpreter(model_file)
        model.allocate_tensors()
        hass.data[DOMAIN][CONF_MODEL] = model

    hass.bus.listen_once(EVENT_HOMEASSISTANT_START, tensorflow_lite_hass_start)

    category_index = load_labels(labels)

    entities = []

    for camera in config[CONF_SOURCE]:
        entities.append(
            TensorFlowLiteImageProcessor(
                hass,
                camera[CONF_ENTITY_ID],
                camera.get(CONF_NAME),
                category_index,
                config,
            )
        )

    add_entities(entities)

class TensorFlowLiteImageProcessor(ImageProcessingEntity):
    """Representation of an TensorFlow Lite image processor."""

    def __init__(
        self,
        hass,
        camera_entity,
        name,
        category_index,
        config,
    ):
        """Initialize the TensorFlow Lite entity."""
        model_config = config.get(CONF_MODEL)
        self.hass = hass
        self._camera_entity = camera_entity
        if name:
            self._name = name
        else:
            self._name = "TensorFlow Lite {}".format(split_entity_id(camera_entity)[1])
        self._category_index = category_index
        self._min_confidence = config.get(CONF_CONFIDENCE)
        self._output = config.get(CONF_FILE_OUT)

        # handle categories and specific detection areas
        self._label_id_offset = model_config.get(CONF_LABEL_OFFSET)
        categories = model_config.get(CONF_CATEGORIES)
        self._include_categories = []
        self._category_areas = {}
        for category in categories:
            if isinstance(category, dict):
                category_name = category.get(CONF_CATEGORY)
                category_area = category.get(CONF_AREA)
                self._include_categories.append(category_name)
                self._category_areas[category_name] = [0, 0, 1, 1]
                if category_area:
                    self._category_areas[category_name] = [
                        category_area.get(CONF_TOP),
                        category_area.get(CONF_LEFT),
                        category_area.get(CONF_BOTTOM),
                        category_area.get(CONF_RIGHT),
                    ]
            else:
                self._include_categories.append(category)
                self._category_areas[category] = [0, 0, 1, 1]

        # Handle global detection area
        self._area = [0, 0, 1, 1]
        area_config = model_config.get(CONF_AREA)
        if area_config:
            self._area = [
                area_config.get(CONF_TOP),
                area_config.get(CONF_LEFT),
                area_config.get(CONF_BOTTOM),
                area_config.get(CONF_RIGHT),
            ]

        template.attach(hass, self._output)

        self._matches = {}
        self._total_matches = 0
        self._process_time = 0

    @property
    def camera_entity(self):
        """Return camera entity id from process pictures."""
        return self._camera_entity

    @property
    def name(self):
        """Return the name of the image processor."""
        return self._name

    @property
    def state(self):
        """Return the state of the entity."""
        return self._total_matches

    @property
    #def extra_state_attributes(self):
    def device_state_attributes(self):
        """Return device specific state attributes."""
        return {
            ATTR_MATCHES: self._matches,
            ATTR_SUMMARY: {
                category: len(values) for category, values in self._matches.items()
            },
            ATTR_TOTAL_MATCHES: self._total_matches,
            ATTR_PROCESS_TIME: self._process_time,
        }

    def _save_image(self, image, matches, paths):
        img = Image.open(io.BytesIO(bytearray(image))).convert("RGB")
        img_width, img_height = img.size
        draw = ImageDraw.Draw(img)

        # Draw custom global region/area
        if self._area != [0, 0, 1, 1]:
            draw_box(
                draw, self._area, img_width, img_height, "Detection Area", (0, 255, 255)
            )

        for category, values in matches.items():
            # Draw custom category regions/areas
            if category in self._category_areas and self._category_areas[category] != [
                0,
                0,
                1,
                1,
            ]:
                label = f"{category.capitalize()} Detection Area"
                draw_box(
                    draw,
                    self._category_areas[category],
                    img_width,
                    img_height,
                    label,
                    (0, 255, 0),
                )

            # Draw detected objects
            for instance in values:
                label = "{} {:.1f}%".format(category, instance["score"])
                draw_box(
                    draw, instance["box"], img_width, img_height, label, (255, 255, 0)
                )

        for path in paths:
            _LOGGER.info("Saving results image to %s", path)
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path), exist_ok=True)
            img.save(path)

    def process_image(self, image):
        """Process the image."""
        model = self.hass.data[DOMAIN][CONF_MODEL]
        if not model:
            _LOGGER.debug("Model not yet ready.")
            return

        start = time.perf_counter()
        
        input_details = model.get_input_details()
        _, input_height, input_width, _ = input_details[0]["shape"]
        
        # Resize image
        try:
            import cv2  # pylint: disable=import-outside-toplevel
            img_resized = cv2.resize(cv2.imdecode(np.asarray(bytearray(image)), cv2.IMREAD_UNCHANGED), (input_width, input_height))
        except ImportError:
            try:
                img_resized = Image.open(io.BytesIO(bytearray(image))).convert("RGB").resize((input_width, input_height), Image.ANTIALIAS)
            except UnidentifiedImageError:
                _LOGGER.warning("Unable to process image, bad data")
                return

        # Set input tensor and invoke
        input_tensor = model.tensor(input_details[0]['index'])
        input_tensor()[0][:, :] = img_resized
        model.invoke()

        # Get all output details
        output_details = model.get_output_details()
        boxes = model.get_tensor(output_details[0]['index'])[0]
        scores = model.get_tensor(output_details[2]['index'])[0]
        classes = (
            model.get_tensor(output_details[1]['index'])[0] + self._label_id_offset
        ).astype(int)

        # Results
        matches = {}
        total_matches = 0
        for box, score, obj_class in zip(boxes, scores, classes):
            score = score * 100
            boxes = box.tolist()

            # Exclude matches below min confidence value
            if score < self._min_confidence:
                continue

            # Exclude matches outside global area definition
            if (
                boxes[0] < self._area[0]
                or boxes[1] < self._area[1]
                or boxes[2] > self._area[2]
                or boxes[3] > self._area[3]
            ):
                continue

            category = self._category_index[obj_class]

            # Exclude unlisted categories
            if self._include_categories and category not in self._include_categories:
                continue

            # Exclude matches outside category specific area definition
            if self._category_areas and (
                boxes[0] < self._category_areas[category][0]
                or boxes[1] < self._category_areas[category][1]
                or boxes[2] > self._category_areas[category][2]
                or boxes[3] > self._category_areas[category][3]
            ):
                continue

            # If we got here, we should include it
            if category not in matches:
                matches[category] = []
            matches[category].append({"score": float(score), "box": boxes})
            total_matches += 1

        # Save Images
        if total_matches and self._output:
            paths = []
            for path_template in self._output:
                if isinstance(path_template, template.Template):
                    paths.append(
                        path_template.render(camera_entity=self._camera_entity)
                    )
                else:
                    paths.append(path_template)
            self._save_image(image, matches, paths)

        self._matches = matches
        self._total_matches = total_matches
        self._process_time = time.perf_counter() - start
