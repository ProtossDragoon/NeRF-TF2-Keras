class RuntimeChecker():
    try:
        import google.colab
        colab_mode = True
    except:
        colab_mode = False


if __name__ == '__main__':
    print(f'COLAB?: {RuntimeChecker.colab_mode}')