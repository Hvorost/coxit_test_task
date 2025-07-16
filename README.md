## Usage
1. Clone the project to your disk.
2. Check the config file and set appropriate parameters for path's. Another parameters you can leave as is or<br />try to play with themand obtain different results.
3. To run the script simply do that ```python detecting_wall_cabinets.py --config_path /your/config/path``` That's it.

## Couple comments
1. The solution is based on template matching technique which is very sensitive from the template quality and size.<br />To get the best result of template matching it is better to take a piece of image (template) from the main image<br />where we're looking for the template.  
2. You can play with different parameters and achieve slightly better or worse results.
3. Parameter values in the config file are optimal don't change them too much.
4. If you run without Canny method set the _confidence_threshold_ to 0.6 ~ 0.7

## Script main logic
1. Find different drawing on the main drawing.
2. Find boxes with wall height template.
3. Combine drawing boxes and height boxes to achieve the result boxes with wall cabinets.
4. Run template matching method for different wall templates on prepared boxes.
