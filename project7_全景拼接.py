import cv2


class PanoramaStitching:
    """
    全景拼接
    """
    def __init__(self):
        images = [cv2.imread(r'./resources/1.jpg'), cv2.imread('./resources/2.jpg'), cv2.imread('./resources/3.jpg')]
        self.images = [cv2.resize(src=img, dsize=None, fx=0.5, fy=0.5) for img in images]

    def run(self):
        stitcher = cv2.Stitcher().create()
        status, pano = stitcher.stitch(self.images)
        if status == cv2.STITCHER_OK:
            cv2.imshow('pano', pano)
            cv2.waitKey(0)


if __name__ == '__main__':
    ps = PanoramaStitching()
    ps.run()

