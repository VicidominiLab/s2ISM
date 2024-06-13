# s²ISM

This python package implements s²ISM (Super-resolution Sectioning Image Scanning Microscopy),
a computational technique to reconstruct images with enhanced resolution, optical sectioning, signal-to-noise ratio
and sampling from a conventional ISM dataset acquired by a laser scanning microscope equipped with a detector array.

The ISM dataset should be a numpy array in the format (x, y, time, channel), where the temporal dimension is not mandatory
and the channel dimension is the flattened 2D dimension of the detector array.

This package also contains a module for simulating instrument-specific PSFs by retrieving the 
relavant parameters automatically from the raw dataset using a minimization procedure.
**Important**: the current implementation of the automatic PSF generation 
works under the assumption that the detectors of the array are arranged in a squared fashion.
If this is not the case for your detector (e.g. AiryScan), you need to provide the PSFs manually.


## Installation

You can install `s2ism` via [pip] directly from GitHub:

    pip install git+https://github.com/VicidominiLab/s2ISM

or using the version on [PyPI]:

    pip install s2ism

It requires the following Python packages

    numpy
    matplotlib
    scipy
    scikit-image
    brighteyes-ism
    torch
    tqdm

## Documentation

You can findexamples of usage here:

https://github.com/VicidominiLab/s2ISM/tree/main/examples

## Citation

If you find s²ISM useful for your research, please cite it as:

_

## License

Distributed under the terms of the [GNU GPL v3.0] license,
"s2ISM" is free and open source software


## Contributing

You want to contribute? Great!
Contributing works best if you creat a pull request with your changes.

1. Fork the project.
2. Create a branch for your feature: `git checkout -b cool-new-feature`
3. Commit your changes: `git commit -am 'My new feature'`
4. Push to the branch: `git push origin cool-new-feature`
5. Submit a pull request!

If you are unfamilar with pull requests, you find more information on pull requests in the
 [github help](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests)

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt

[file an issue]: https://github.com/VicidominiLab/s2ism/issues

[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/project/s2ism/


