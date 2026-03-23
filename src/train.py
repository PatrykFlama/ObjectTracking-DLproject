from RF_DETR import RFDETRT

if __name__ == "__main__":
    rfdetr = RFDETRT("nano")
    rfdetr.train("/Users/piotrek/ObjectTracking-DLproject/dataset",
                 epochs=10, batch_size=4)
