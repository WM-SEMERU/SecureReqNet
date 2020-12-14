"""gamma_data dataset."""

import tensorflow_datasets as tfds
import os
import tensorflow as tf

# TODO(gamma_data): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(gamma_data): BibTeX citation
_CITATION = """
"""

def _label(string):
    if string == "(0,1)":
        return [0,1]
    else:
        return [1,0]
    
class GammaData(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for gamma_data dataset."""

  VERSION = tfds.core.Version('1.0.7')
  RELEASE_NOTES = {
      '1.0.7': 'Full data with text instead of sentence',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(gamma_data): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'text': tfds.features.Text(),
            'label': tfds.features.ClassLabel(names=["(0,1)","(1,0)"]),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=("text", "label"),  # e.g. ('image', 'label')
        homepage='',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(gamma_data): Downloads the data and defines the splits
    data_set = "augmented_dataset"
    path = os.path.join("..","data",data_set)
    train_path = os.path.join(path,"full_ground_truth_train.txt")
    test_path = os.path.join(path,"full_ground_truth_test.txt")

    # TODO(gamma_data): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(train_path,path),
        'test': self._generate_examples(test_path,path),
        
    }

  def _generate_examples(self, ground_path, data_path):
    """Yields examples."""
    # TODO(gamma_data): Yields (key, example) tuples from the dataset
    data_set = "augmented_dataset"
    f = open(ground_path, "r")
    for row in f:
        row = row.split(" ")
        text_name = row[0]
        file = open(os.path.join(data_path,data_set,"issues",text_name),"r")
        sentence = file.readlines()
        file.close()
        # And yield (key, feature_dict)
        yield text_name, {
            'text': " ".join(sentence),
            'label': row[1].rstrip(),
      }
            'sentence': " ".join(sentence),
            'label': row[1].rstrip(),
      }

