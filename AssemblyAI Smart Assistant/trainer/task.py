from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_string("caption", None, "caption to generate an image for")

def main(_argv):

    """
    Run the model and generate an image of FLAGS.caption as the prompt

    Then save the resulting image to a GCP bucket
    """




if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass


