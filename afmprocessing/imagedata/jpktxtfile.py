import numpy as np

def read_afm_jpk_txt_file(file_path):
	if not file_path.lower().endswith('.txt'):
		raise ValueError("Input must be a jpk text file with .txt extension")
		
	try:
		metadata = {}
		height_data = []

		with open(file_path, 'r') as file:
			for line in file:
				line = line.strip()
				if line.startswith('#'):
					# Process metadata
					if ':' in line:
						key, value = line[1:].split(':', 1)
						metadata[key.strip()] = value.strip()
				elif line:
					# Process numerical data (i.e., height values)
					height_data.append([float(x) for x in line.split()])

		return metadata, np.array(height_data)

	except IOError:
		raise IOError(f"Unable to read the file: {file_path}")
	except UnicodeDecodeError:
		raise ValueError("The file is not a valid text file or contains non-text content")
		
def create_height_image(height_data):
	# Find the minimum and maximum values in the image
    minVal = height_data.min()
    maxVal = height_data.max()

    # % Normalize the image to the range [0, 255]
    return np.uint8((height_data - minVal) / (maxVal - minVal)  * 255);


def save_AFM_height_data(height_data, file_path, original_metadata=None):
	if not file_path.lower().endswith('.txt'):
		raise ValueError("The file must be a text file with .txt extension")
	try:
		with open(file_path, 'w') as file:
			# Write metadata
			if original_metadata:
				for key, value in original_metadata.items():
					file.write(f"# {key}: {value}\n")
			else: 
				# Save into format so that it can be open by Gwyddion.
				# Change the metadata which suits to AFM data you are investigated 
				# If no original metadata, write some basic information
				file.write(f"# Channel: Height (flattened)\n")
				file.write(f"# Width: 10.00000000000001 um\n")
				file.write(f"# Height: 10.00000000000001 um\n")

				file.write(f"# Value units: m\n")
			
			# Write numerical data
			for row in height_data:
				file.write(' '.join(f"{value:.16e}" for value in row) + '\n')
	except IOError:
		raise IOError(f"Unable to read the file: {file_path}")
	except UnicodeDecodeError:
		raise ValueError("The file is not a valid text file or contains non-text content")