^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package movie_publisher
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1.1.2 (2019-03-07)
------------------
* Made imageio and moviepy mandatory dependencies (they will be removed from package.xml in release repo)
* Contributors: Martin Pecka

1.1.1 (2019-02-07)
------------------
* Fixed permissions.
* Moved to python from bc, because it is not installed everywhere.
* More informative error strings.
* Updated to the fixed version rosbash_params==1.0.2.
* Contributors: Martin Pecka

1.1.0 (2019-01-28)
------------------
* Added checks for exit codes to bash scripts.
* Fixed install targets.
* Added python-opencv alternative backend. This resolves debian packaging issues.
* Contributors: Martin Pecka

1.0.1 (2019-01-25)
------------------
* Fixed install rule.
* Documented all tools in readme.
* Added support for rewriting timestamps from TF messages.
* Fix Python3 compatibility. Added fix_bag_timestamps.
* Added readme.
* Initial commit.
* Contributors: Martin Pecka
