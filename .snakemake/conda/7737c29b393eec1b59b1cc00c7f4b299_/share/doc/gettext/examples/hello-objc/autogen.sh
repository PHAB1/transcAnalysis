#!/bin/sh
# Example for use of GNU gettext.
# This file is in the public domain.
#
# Script for regenerating all autogenerated files.

if test -r ../Makefile.am; then
  # Inside the gettext source directory.
  GETTEXT_TOPSRCDIR=../../..
else
  if test -r ../Makefile; then
    # Inside a gettext build directory.
    GETTEXT_TOOLS_SRCDIR=`sed -n -e 's,^top_srcdir *= *\(.*\)$,\1,p' ../Makefile`
    # Adjust a relative top_srcdir.
    case $GETTEXT_TOOLS_SRCDIR in
      /*) ;;
      *) GETTEXT_TOOLS_SRCDIR=../$GETTEXT_TOOLS_SRCDIR ;;
    esac
    GETTEXT_TOPSRCDIR=$GETTEXT_TOOLS_SRCDIR/../..
  else
    # Installed under ${prefix}/share/doc/gettext/examples.
    . ../installpaths
  fi
fi

cp -p ${GETTEXTSRCDIR-$GETTEXT_TOPSRCDIR/gettext-tools/gnulib-lib}/gettext.h gettext.h

autopoint -f # was: gettextize -f -c
rm po/Makevars.template
rm po/Rules-quot
rm po/boldquot.sed
rm po/en@boldquot.header
rm po/en@quot.header
rm po/insert-header.sin
rm po/quot.sed

aclocal -I m4

autoconf

automake -a -c

cd po
for f in *.po; do
  if test -r "$f"; then
    lang=`echo $f | sed -e 's,\.po$,,'`
    msgfmt -c -o $lang.gmo $lang.po
  fi
done
cd ..