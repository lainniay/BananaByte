Detect two parts: main object and the possible reflective layer(possibly a quadrilateral area and takes only part of the image)
Output a JSON list of objects. Each object must contain: "
            "'box_2d' [ymin, xmin, ymax, xmax] (normalized 0-1000) and 'label'. "
            "Do not include any mask data."