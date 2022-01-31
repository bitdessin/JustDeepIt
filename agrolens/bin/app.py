import argparse


def main():
    parser = argparse.ArgumentParser(description='AgroLens')  
    parser.add_argument('app', help='Set `OD` to start application for object detection, `IS` to start application for instance  segmentation, `SOD` to start application for saslient object detection.')
    args = parser.parse_args()
    
    app = None
    if args.app.upper() == 'OD':
        from agrolens import ODGUI
        app = ODGUI()
    elif args.app.upper() == 'IS':
        from agrolens import ISGUI
        app = ISGUI()
    elif args.app.upper() == 'SOD':
        from agrolens import SODGUI
        app = SODGUI()
    else:
        raise NotImplementedError
    
    app.startup()


if __name__ == '__main__':
    main()

