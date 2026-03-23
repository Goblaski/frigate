#!/bin/bash

set -euxo pipefail

hailo_variant="${HAILO_VARIANT:-hailo8}"
hailo_version="${HAILO_VERSION:-4.21.0}"
hailo_tappas_core_version="${HAILO_TAPPAS_CORE_VERSION:-}"
rpi_suite="${HAILO_RPI_SUITE:-trixie}"

if [[ "${TARGETARCH}" == "amd64" ]]; then
    arch="x86_64"
elif [[ "${TARGETARCH}" == "arm64" ]]; then
    arch="aarch64"
else
    echo "Unsupported TARGETARCH: ${TARGETARCH}"
    exit 1
fi

install_legacy_hailort() {
    wget -qO- "https://github.com/frigate-nvr/hailort/releases/download/v${hailo_version}/hailort-debian12-${TARGETARCH}.tar.gz" | tar -C / -xzf -
    wget -P /wheels/ "https://github.com/frigate-nvr/hailort/releases/download/v${hailo_version}/hailort-${hailo_version}-cp311-cp311-linux_${arch}.whl"
}

resolve_deb_version() {
    local package_name="$1"
    local version_prefix="$2"

    if [[ -z "${version_prefix}" ]]; then
        return 0
    fi

    local resolved_version
    resolved_version=$(apt-cache madison "${package_name}" | awk '{print $3}' | grep -E "^${version_prefix}([.+~-].*)?$" | head -n1 || true)

    if [[ -z "${resolved_version}" ]]; then
        echo "Unable to resolve ${package_name} version prefix '${version_prefix}'." >&2
        echo "Available versions:" >&2
        apt-cache madison "${package_name}" >&2 || true
        exit 1
    fi

    printf '%s' "${resolved_version}"
}


install_hailo10h_userspace() {
    if [[ "${TARGETARCH}" != "arm64" ]]; then
        echo "hailo10h userspace installation is only supported on arm64 builds."
        exit 1
    fi

    apt-get -qq update
    apt-get -qq install --no-install-recommends -y \
        apt-transport-https \
        ca-certificates \
        curl \
        gnupg

    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://archive.raspberrypi.com/debian/raspberrypi.gpg.key | \
        gpg --dearmor -o /etc/apt/keyrings/raspberrypi.gpg
    chmod a+r /etc/apt/keyrings/raspberrypi.gpg

    cat >/etc/apt/sources.list.d/raspberrypi.sources <<EOF
Types: deb
URIs: https://archive.raspberrypi.com/debian/
Suites: ${rpi_suite}
Components: main
Signed-By: /etc/apt/keyrings/raspberrypi.gpg
EOF

    apt-get -qq update

    mkdir -p /rootfs /tmp/hailo-debs
    cd /tmp/hailo-debs

    hailort_deb_version=$(resolve_deb_version "hailort" "${hailo_version}")
    python_hailort_deb_version=$(resolve_deb_version "python3-hailort" "${hailo_version}")

    packages=(
        "hailort=${hailort_deb_version}"
        "python3-hailort=${python_hailort_deb_version}"
    )

    if [[ -n "${hailo_tappas_core_version}" ]]; then
        tappas_core_deb_version=$(resolve_deb_version "hailo-tappas-core" "${hailo_tappas_core_version}")
        packages+=("hailo-tappas-core=${tappas_core_deb_version}")
    fi

    for package in "${packages[@]}"; do
        apt-get download "${package}"
    done

    for deb in ./*.deb; do
        dpkg-deb -x "${deb}" /rootfs
    done

    rm -rf /tmp/hailo-debs
}

case "${hailo_variant}" in
    hailo10h)
        install_hailo10h_userspace
        ;;
    hailo8|legacy)
        install_legacy_hailort
        ;;
    *)
        echo "Unknown HAILO_VARIANT: ${hailo_variant}"
        exit 1
        ;;
esac
