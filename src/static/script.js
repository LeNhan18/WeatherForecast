function fetchForecast() {
    const table = document.getElementById('forecastTable');
    const loading = document.getElementById('loading');
    table.style.display = 'none';
    loading.style.display = 'block';

    fetch('/api/forecast')
        .then(response => {
            if (!response.ok) {
                throw new Error('Lỗi khi gọi API: ' + response.status);
            }
            return response.json();
        })
        .then(data => {
            console.log('Dữ liệu nhận được:', data);
            const tableBody = document.getElementById('forecastBody');
            tableBody.innerHTML = '';

            if (data.length === 0) {
                alert('Không có dữ liệu dự báo!');
                return;
            }

            data.forEach(item => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${item.day}</td>
                    <td>${item.predicted}</td>
                    <td>${item.actual}</td>
                `;
                tableBody.appendChild(row);
            });
            loading.style.display = 'none';
            table.style.display = 'table';
        })
        .catch(error => {
            console.error('Lỗi:', error);
            alert('Có lỗi xảy ra khi lấy dự báo: ' + error.message);
            loading.style.display = 'none';
        });
}