#!/usr/bin/env bash

this="${BASH_SOURCE-$0}"
while [ -h "$this" ]; do
    ls=`ls -ld "$this"`
    link=`expr "$ls" : '.*-> \(.*\)$'`
    if expr "$link" : '.*/.*' > /dev/null; then
	this="$link"
    else
	this=`dirname "$this"`/"$link"
    fi
done

# convert relative path to absolute path
bin=`dirname "$this"`
script=`basename "$this"`
bin=`cd "$bin">/dev/null; pwd`
this="$bin/$script"

if [ -z "$APP_HOME" ]; then
    export APP_HOME=`dirname "$this"`/..
fi

APP_CONF_DIR="${APP_CONF_DIR:-$APP_HOME/conf}"

if [ -f "$APP_CONF_DIR/env.sh" ]; then
    . "$APP_CONF_DIR/env.sh"
fi

if [ -n "$CONDA_ENV_NAME" ]; then
    echo "Activating conda env $CONDA_ENV_NAME"
    eval "$(conda shell.bash hook)"
    conda activate $CONDA_ENV_NAME
fi

if type -p python; then
    echo Found Python executable in PATH
    PYTHON="python"
else
    cat 1>&2 <<EOF
+======================================================================+
|                Error: Python could not be found                      |
+----------------------------------------------------------------------+
| Please install Python or conda                                       |
+======================================================================+
EOF
    exit 1
fi

if [ -z ${APP_MAIN_FILE} ]; then
    echo "You must set app main class name in conf/env.sh"
    exit 1
fi

exec ${PYTHON} ${APP_OPTS} -m ${APP_MAIN_FILE} $@