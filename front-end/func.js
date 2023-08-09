var BACK_END = "https://your.backend.server.cn/api";

function queryHTTP(url) {
    const xhr = new XMLHttpRequest();
    xhr.open('get', url, false);
    xhr.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
    xhr.send(null);
    return xhr.responseText
}
function getQueryString(name) {
    let reg = new RegExp("(^|&)" + name + "=([^&]*)(&|$)", "i");
    let r = window.location.search.substr(1).match(reg);
    if (r != null) {
        return decodeURIComponent(r[2]);
    };
    return null;
}
