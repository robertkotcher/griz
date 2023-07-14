This repo implements an application that lets you chat with your data.

# build

please note that I follow a very janky process to build app for deployment

1. `npm run build` in `/webapp`
2. `mv webapp/build static`
3. `mv static/static/js static/js`
4. `mv static/static/css static/css`

